# LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation 

**Authors**: Wangyu Wu, Zhenhong Chen, Xianglin Qiu, Siqi Song, Xiaowei Huang, Fei Ma, Jimin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17966)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences and capturing both intra- and inter-sequence item relationships. We propose LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation (LLM-EMF), a novel and advanced approach that enhances textual information with Large Language Models (LLM) knowledge and significantly improves recommendation performance through the fusion of visual and textual data. Using the frozen CLIP model, we generate image and text embeddings, thereby enriching item representations with multimodal data. A multiple attention mechanism jointly learns both single-domain and cross-domain preferences, effectively capturing and understanding complex user interests across diverse domains. Evaluations conducted on four e-commerce datasets demonstrate that LLM-EMF consistently outperforms existing methods in modeling cross-domain user preferences, thereby highlighting the effectiveness of multimodal data integration and its advantages in enhancing sequential recommendation systems. Our source code will be released. 

---
# Enhancing Document Retrieval in COVID-19 Research: Leveraging Large Language Models for Hidden Relation Extraction 

**Authors**: Hoang-An Trieu, Dinh-Truong Do, Chau Nguyen, Vu Tran, Minh Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18311)  

**Abstract**: In recent years, with the appearance of the COVID-19 pandemic, numerous publications relevant to this disease have been issued. Because of the massive volume of publications, an efficient retrieval system is necessary to provide researchers with useful information if an unexpected pandemic happens so suddenly, like COVID-19. In this work, we present a method to help the retrieval system, the Covrelex-SE system, to provide more high-quality search results. We exploited the power of the large language models (LLMs) to extract the hidden relationships inside the unlabeled publication that cannot be found by the current parsing tools that the system is using. Since then, help the system to have more useful information during retrieval progress. 

---
# LettinGo: Explore User Profile Generation for Recommendation System 

**Authors**: Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18309)  

**Abstract**: User profiling is pivotal for recommendation systems, as it transforms raw user interaction data into concise and structured representations that drive personalized recommendations. While traditional embedding-based profiles lack interpretability and adaptability, recent advances with large language models (LLMs) enable text-based profiles that are semantically richer and more transparent. However, existing methods often adhere to fixed formats that limit their ability to capture the full diversity of user behaviors. In this paper, we introduce LettinGo, a novel framework for generating diverse and adaptive user profiles. By leveraging the expressive power of LLMs and incorporating direct feedback from downstream recommendation tasks, our approach avoids the rigid constraints imposed by supervised fine-tuning (SFT). Instead, we employ Direct Preference Optimization (DPO) to align the profile generator with task-specific performance, ensuring that the profiles remain adaptive and effective. LettinGo operates in three stages: (1) exploring diverse user profiles via multiple LLMs, (2) evaluating profile quality based on their impact in recommendation systems, and (3) aligning the profile generation through pairwise preference data derived from task performance. Experimental results demonstrate that our framework significantly enhances recommendation accuracy, flexibility, and contextual awareness. This work enhances profile generation as a key innovation for next-generation recommendation systems. 

---
# Context-Aware Scientific Knowledge Extraction on Linked Open Data using Large Language Models 

**Authors**: Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17580)  

**Abstract**: The exponential growth of scientific literature challenges researchers extracting and synthesizing knowledge. Traditional search engines return many sources without direct, detailed answers, while general-purpose LLMs may offer concise responses that lack depth or omit current information. LLMs with search capabilities are also limited by context window, yielding short, incomplete answers. This paper introduces WISE (Workflow for Intelligent Scientific Knowledge Extraction), a system addressing these limits by using a structured workflow to extract, refine, and rank query-specific knowledge. WISE uses an LLM-powered, tree-based architecture to refine data, focusing on query-aligned, context-aware, and non-redundant information. Dynamic scoring and ranking prioritize unique contributions from each source, and adaptive stopping criteria minimize processing overhead. WISE delivers detailed, organized answers by systematically exploring and synthesizing knowledge from diverse sources. Experiments on HBB gene-associated diseases demonstrate WISE reduces processed text by over 80% while achieving significantly higher recall over baselines like search engines and other LLM-based approaches. ROUGE and BLEU metrics reveal WISE's output is more unique than other systems, and a novel level-based metric shows it provides more in-depth information. We also explore how the WISE workflow can be adapted for diverse domains like drug discovery, material science, and social science, enabling efficient knowledge extraction and synthesis from unstructured scientific papers and web sources. 

---
# Harnessing the Power of Reinforcement Learning for Language-Model-Based Information Retriever via Query-Document Co-Augmentation 

**Authors**: Jingming Liu, Yumeng Li, Wei Shi, Yao-Xiang Ding, Hui Su, Kun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18670)  

**Abstract**: Recent studies have proposed leveraging Large Language Models (LLMs) as information retrievers through query rewriting. However, for challenging corpora, we argue that enhancing queries alone is insufficient for robust semantic matching; the LLM should also have sufficient understanding of the corpus by directly handling and augmenting the documents themselves. To this end, we present an LLM-based retriever empowered to augment both user queries and corpus documents, with its policy fully explored via reinforcement learning (RL) and minimal human inductive bias. Notably, we find that simply allowing the LLM to modify documents yields little benefit unless paired with our carefully designed bidirectional RL framework, which enables the LLM to simultaneously learn and collaborate on both query and document augmentation policies. A key technical challenge in realizing such a framework lies in jointly updating both policies during training, where the rewards for the two directions depend on each other, making their entangled reward intractable. Our approach addresses this by introducing a reward sampling strategy and a specifically designed RL algorithm that enables effective training with these sampled rewards. Experimental results demonstrate that our approach significantly enhances LLM-based retrieval performance in both sparse and dense settings, particularly in difficult retrieval domains, and achieves strong cross-benchmark generalization. Our code is released at this https URL. 

---
# Expanding Relevance Judgments for Medical Case-based Retrieval Task with Multimodal LLMs 

**Authors**: Catarina Pires, Sérgio Nunes, Luís Filipe Teixeira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17782)  

**Abstract**: Evaluating Information Retrieval (IR) systems relies on high-quality manual relevance judgments (qrels), which are costly and time-consuming to obtain. While pooling reduces the annotation effort, it results in only partially labeled datasets. Large Language Models (LLMs) offer a promising alternative to reducing reliance on manual judgments, particularly in complex domains like medical case-based retrieval, where relevance assessment requires analyzing both textual and visual information. In this work, we explore using a Multimodal Large Language Model (MLLM) to expand relevance judgments, creating a new dataset of automated judgments. Specifically, we employ Gemini 1.5 Pro on the ImageCLEFmed 2013 case-based retrieval task, simulating human assessment through an iteratively refined, structured prompting strategy that integrates binary scoring, instruction-based evaluation, and few-shot learning. We systematically experimented with various prompt configurations to maximize agreement with human judgments. To evaluate agreement between the MLLM and human judgments, we use Cohen's Kappa, achieving a substantial agreement score of 0.6, comparable to inter-annotator agreement typically observed in multimodal retrieval tasks. Starting from the original 15,028 manual judgments (4.72% relevant) across 35 topics, our MLLM-based approach expanded the dataset by over 37x to 558,653 judgments, increasing relevant annotations to 5,950. On average, each medical case query received 15,398 new annotations, with approximately 99% being non-relevant, reflecting the high sparsity typical in this domain. Our results demonstrate the potential of MLLMs to scale relevance judgment collection, offering a promising direction for supporting retrieval evaluation in medical and multimodal IR tasks. 

---
# A Framework for Generating Conversational Recommendation Datasets from Behavioral Interactions 

**Authors**: Vinaik Chhetri, Yousaf Reza, Moghis Fereidouni, Srijata Maji, Umar Farooq, AB Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2506.17285)  

**Abstract**: Modern recommendation systems typically follow two complementary paradigms: collaborative filtering, which models long-term user preferences from historical interactions, and conversational recommendation systems (CRS), which interact with users in natural language to uncover immediate needs. Each captures a different dimension of user intent. While CRS models lack collaborative signals, leading to generic or poorly personalized suggestions, traditional recommenders lack mechanisms to interactively elicit immediate needs. Unifying these paradigms promises richer personalization but remains challenging due to the lack of large-scale conversational datasets grounded in real user behavior. We present ConvRecStudio, a framework that uses large language models (LLMs) to simulate realistic, multi-turn dialogs grounded in timestamped user-item interactions and reviews. ConvRecStudio follows a three-stage pipeline: (1) Temporal Profiling, which constructs user profiles and community-level item sentiment trajectories over fine-grained aspects; (2) Semantic Dialog Planning, which generates a structured plan using a DAG of flexible super-nodes; and (3) Multi-Turn Simulation, which instantiates the plan using paired LLM agents for the user and system, constrained by executional and behavioral fidelity checks. We apply ConvRecStudio to three domains -- MobileRec, Yelp, and Amazon Electronics -- producing over 12K multi-turn dialogs per dataset. Human and automatic evaluations confirm the naturalness, coherence, and behavioral grounding of the generated conversations. To demonstrate utility, we build a cross-attention transformer model that jointly encodes user history and dialog context, achieving gains in Hit@K and NDCG@K over baselines using either signal alone or naive fusion. Notably, our model achieves a 10.9% improvement in Hit@1 on Yelp over the strongest baseline. 

---
# Automating Financial Statement Audits with Large Language Models 

**Authors**: Rushi Wang, Jiateng Liu, Weijie Zhao, Shenglan Li, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17282)  

**Abstract**: Financial statement auditing is essential for stakeholders to understand a company's financial health, yet current manual processes are inefficient and error-prone. Even with extensive verification procedures, auditors frequently miss errors, leading to inaccurate financial statements that fail to meet stakeholder expectations for transparency and reliability. To this end, we harness large language models (LLMs) to automate financial statement auditing and rigorously assess their capabilities, providing insights on their performance boundaries in the scenario of automated auditing. Our work introduces a comprehensive benchmark using a curated dataset combining real-world financial tables with synthesized transaction data. In the benchmark, we developed a rigorous five-stage evaluation framework to assess LLMs' auditing capabilities. The benchmark also challenges models to map specific financial statement errors to corresponding violations of accounting standards, simulating real-world auditing scenarios through test cases. Our testing reveals that current state-of-the-art LLMs successfully identify financial statement errors when given historical transaction data. However, these models demonstrate significant limitations in explaining detected errors and citing relevant accounting standards. Furthermore, LLMs struggle to execute complete audits and make necessary financial statement revisions. These findings highlight a critical gap in LLMs' domain-specific accounting knowledge. Future research must focus on enhancing LLMs' understanding of auditing principles and procedures. Our benchmark and evaluation framework establish a foundation for developing more effective automated auditing tools that will substantially improve the accuracy and efficiency of real-world financial statement auditing. 

---
# Team LA at SCIDOCA shared task 2025: Citation Discovery via relation-based zero-shot retrieval 

**Authors**: Trieu An, Long Nguyen, Minh Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18316)  

**Abstract**: The Citation Discovery Shared Task focuses on predicting the correct citation from a given candidate pool for a given paragraph. The main challenges stem from the length of the abstract paragraphs and the high similarity among candidate abstracts, making it difficult to determine the exact paper to cite. To address this, we develop a system that first retrieves the top-k most similar abstracts based on extracted relational features from the given paragraph. From this subset, we leverage a Large Language Model (LLM) to accurately identify the most relevant citation. We evaluate our framework on the training dataset provided by the SCIDOCA 2025 organizers, demonstrating its effectiveness in citation prediction. 

---
# Mapping the Evolution of Research Contributions using KnoVo 

**Authors**: Sajratul Y. Rubaiat, Syed N. Sakib, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17508)  

**Abstract**: This paper presents KnoVo (Knowledge Evolution), an intelligent framework designed for quantifying and analyzing the evolution of research novelty in the scientific literature. Moving beyond traditional citation analysis, which primarily measures impact, KnoVo determines a paper's novelty relative to both prior and subsequent work within its multilayered citation network. Given a target paper's abstract, KnoVo utilizes Large Language Models (LLMs) to dynamically extract dimensions of comparison (e.g., methodology, application, dataset). The target paper is then compared to related publications along these same extracted dimensions. This comparative analysis, inspired by tournament selection, yields quantitative novelty scores reflecting the relative improvement, equivalence, or inferiority of the target paper in specific aspects. By aggregating these scores and visualizing their progression, for instance, through dynamic evolution graphs and comparative radar charts, KnoVo facilitates researchers not only to assess originality and identify similar work, but also to track knowledge evolution along specific research dimensions, uncover research gaps, and explore cross-disciplinary connections. We demonstrate these capabilities through a detailed analysis of 20 diverse papers from multiple scientific fields and report on the performance of various open-source LLMs within the KnoVo framework. 

---
# CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models 

**Authors**: Junze Chen, Xinjie Yang, Cheng Yang, Junfei Bao, Zeyuan Guo, Yawen Li, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17281)  

**Abstract**: Recommender systems (RSs) are designed to retrieve candidate items a user might be interested in from a large pool. A common approach is using graph neural networks (GNNs) to capture high-order interaction relationships. As large language models (LLMs) have shown strong capabilities across domains, researchers are exploring their use to enhance recommendation. However, prior work limits LLMs to re-ranking results or dataset augmentation, failing to utilize their power during candidate filtering - which may lead to suboptimal performance. Instead, we propose to leverage LLMs' reasoning abilities during the candidate filtering process, and introduce Chain Of Retrieval ON grAphs (CORONA) to progressively narrow down the range of candidate items on interaction graphs with the help of LLMs: (1) First, LLM performs preference reasoning based on user profiles, with the response serving as a query to extract relevant users and items from the interaction graph as preference-assisted retrieval; (2) Then, using the information retrieved in the previous step along with the purchase history of target user, LLM conducts intent reasoning to help refine an even smaller interaction subgraph as intent-assisted retrieval; (3) Finally, we employ a GNN to capture high-order collaborative filtering information from the extracted subgraph, performing GNN-enhanced retrieval to generate the final recommendation results. The proposed framework leverages the reasoning capabilities of LLMs during the retrieval process, while seamlessly integrating GNNs to enhance overall recommendation performance. Extensive experiments on various datasets and settings demonstrate that our proposed CORONA achieves state-of-the-art performance with an 18.6% relative improvement in recall and an 18.4% relative improvement in NDCG on average. 

---
# CARTS: Collaborative Agents for Recommendation Textual Summarization 

**Authors**: Jiao Chen, Kehui Yao, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17765)  

**Abstract**: Current recommendation systems often require some form of textual data summarization, such as generating concise and coherent titles for product carousels or other grouped item displays. While large language models have shown promise in NLP domains for textual summarization, these approaches do not directly apply to recommendation systems, where explanations must be highly relevant to the core features of item sets, adhere to strict word limit constraints. In this paper, we propose CARTS (Collaborative Agents for Recommendation Textual Summarization), a multi-agent LLM framework designed for structured summarization in recommendation systems. CARTS decomposes the task into three stages-Generation Augmented Generation (GAG), refinement circle, and arbitration, where successive agent roles are responsible for extracting salient item features, iteratively refining candidate titles based on relevance and length feedback, and selecting the final title through a collaborative arbitration process. Experiments on large-scale e-commerce data and live A/B testing show that CARTS significantly outperforms single-pass and chain-of-thought LLM baselines, delivering higher title relevance and improved user engagement metrics. 

---
# ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs 

**Authors**: Jiaru Zou, Ling Yang, Jingwen Gu, Jiahao Qiu, Ke Shen, Jingrui He, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18896)  

**Abstract**: Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Projects: this https URL 

---
# ASP2LJ : An Adversarial Self-Play Laywer Augmented Legal Judgment Framework 

**Authors**: Ao Chang, Tong Zhou, Yubo Chen, Delai Qiu, Shengping Liu, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18768)  

**Abstract**: Legal Judgment Prediction (LJP) aims to predict judicial outcomes, including relevant legal charge, terms, and fines, which is a crucial process in Large Language Model(LLM). However, LJP faces two key challenges: (1)Long Tail Distribution: Current datasets, derived from authentic cases, suffer from high human annotation costs and imbalanced distributions, leading to model performance degradation. (2)Lawyer's Improvement: Existing systems focus on enhancing judges' decision-making but neglect the critical role of lawyers in refining arguments, which limits overall judicial accuracy. To address these issues, we propose an Adversarial Self-Play Lawyer Augmented Legal Judgment Framework, called ASP2LJ, which integrates a case generation module to tackle long-tailed data distributions and an adversarial self-play mechanism to enhance lawyers' argumentation skills. Our framework enables a judge to reference evolved lawyers' arguments, improving the objectivity, fairness, and rationality of judicial decisions. Besides, We also introduce RareCases, a dataset for rare legal cases in China, which contains 120 tail-end cases. We demonstrate the effectiveness of our approach on the SimuCourt dataset and our RareCases dataset. Experimental results show our framework brings improvements, indicating its utilization. Our contributions include an integrated framework, a rare-case dataset, and publicly releasing datasets and code to support further research in automated judicial systems. 

---
# OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization 

**Authors**: Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.18880)  

**Abstract**: Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency. 

---
# RWESummary: A Framework and Test for Choosing Large Language Models to Summarize Real-World Evidence (RWE) Studies 

**Authors**: Arjun Mukerji, Michael L. Jackson, Jason Jones, Neil Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18819)  

**Abstract**: Large Language Models (LLMs) have been extensively evaluated for general summarization tasks as well as medical research assistance, but they have not been specifically evaluated for the task of summarizing real-world evidence (RWE) from structured output of RWE studies. We introduce RWESummary, a proposed addition to the MedHELM framework (Bedi, Cui, Fuentes, Unell et al., 2025) to enable benchmarking of LLMs for this task. RWESummary includes one scenario and three evaluations covering major types of errors observed in summarization of medical research studies and was developed using Atropos Health proprietary data. Additionally, we use RWESummary to compare the performance of different LLMs in our internal RWE summarization tool. At the time of publication, with 13 distinct RWE studies, we found the Gemini 2.5 models performed best overall (both Flash and Pro). We suggest RWESummary as a novel and useful foundation model benchmark for real-world evidence study summarization. 

---
# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18841)  

**Abstract**: Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL 

---
# Is There a Case for Conversation Optimized Tokenizers in Large Language Models? 

**Authors**: Raquel Ferrando, Javier Conde, Gonzalo Martínez, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2506.18674)  

**Abstract**: The computational and energy costs of Large Language Models (LLMs) have increased exponentially driven by the growing model sizes and the massive adoption of LLMs by hundreds of millions of users. The unit cost of an LLM is the computation of a token. Therefore, the tokenizer plays an important role in the efficiency of a model, and they are carefully optimized to minimize the number of tokens for the text in their training corpus. One of the most popular applications of LLMs are chatbots that interact with users. A key observation is that, for those chatbots, what is important is the performance of the tokenizer in the user text input and the chatbot responses. Those are most likely different from the text in the training corpus. So, a question that immediately arises is whether there is a potential benefit in optimizing tokenizers for chatbot conversations. In this paper, this idea is explored for different tokenizers by using a publicly available corpus of chatbot conversations to redesign their vocabularies and evaluate their performance in this domain. The results show that conversation-optimized tokenizers consistently reduce the number of tokens in chatbot dialogues, which can lead to meaningful energy savings, in the range of 5% to 10% while having minimal or even slightly positive impact on tokenization efficiency for the original training corpus. 

---
# STU-PID: Steering Token Usage via PID Controller for Efficient Large Language Model Reasoning 

**Authors**: Aryasomayajula Ram Bharadwaj  

**Link**: [PDF](https://arxiv.org/pdf/2506.18831)  

**Abstract**: Large Language Models employing extended chain-of-thought (CoT) reasoning often suffer from the overthinking phenomenon, generating excessive and redundant reasoning steps that increase computational costs while potentially degrading performance. While recent work has explored static steering approaches to mitigate this issue, they lack the adaptability to dynamically adjust intervention strength based on real-time reasoning quality. We propose STUPID (Steering Token Usage via PID controller), a novel training-free method that employs a PID controller to dynamically modulate activation steering strength during inference. Our approach combines a chunk-level classifier for detecting redundant reasoning patterns with a PID control mechanism that adaptively adjusts steering intensity based on the predicted redundancy probability. Experimental evaluation on GSM8K demonstrates that STUPID achieves a 6% improvement in accuracy while reducing token usage by 32%, outperforming static steering baselines. Our method provides a principled framework for dynamic reasoning calibration that maintains reasoning quality while significantly improving computational efficiency. 

---
# Comparative Evaluation of ChatGPT and DeepSeek Across Key NLP Tasks: Strengths, Weaknesses, and Domain-Specific Performance 

**Authors**: Wael Etaiwi, Bushra Alhijawi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18501)  

**Abstract**: The increasing use of large language models (LLMs) in natural language processing (NLP) tasks has sparked significant interest in evaluating their effectiveness across diverse applications. While models like ChatGPT and DeepSeek have shown strong results in many NLP domains, a comprehensive evaluation is needed to understand their strengths, weaknesses, and domain-specific abilities. This is critical as these models are applied to various tasks, from sentiment analysis to more nuanced tasks like textual entailment and translation. This study aims to evaluate ChatGPT and DeepSeek across five key NLP tasks: sentiment analysis, topic classification, text summarization, machine translation, and textual entailment. A structured experimental protocol is used to ensure fairness and minimize variability. Both models are tested with identical, neutral prompts and evaluated on two benchmark datasets per task, covering domains like news, reviews, and formal/informal texts. The results show that DeepSeek excels in classification stability and logical reasoning, while ChatGPT performs better in tasks requiring nuanced understanding and flexibility. These findings provide valuable insights for selecting the appropriate LLM based on task requirements. 

---
# TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models 

**Authors**: Ce Li, Xiaofan Liu, Zhiyan Song, Ce Chi, Chen Zhao, Jingjing Yang, Zhendong Wang, Kexin Yang, Boshen Shi, Xing Wang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18421)  

**Abstract**: The majority of data in businesses and industries is stored in tables, databases, and data warehouses. Reasoning with table-structured data poses significant challenges for large language models (LLMs) due to its hidden semantics, inherent complexity, and structured nature. One of these challenges is lacking an effective evaluation benchmark fairly reflecting the performances of LLMs on broad table reasoning abilities. In this paper, we fill in this gap, presenting a comprehensive table reasoning evolution benchmark, TReB, which measures both shallow table understanding abilities and deep table reasoning abilities, a total of 26 sub-tasks. We construct a high quality dataset through an iterative data processing procedure. We create an evaluation framework to robustly measure table reasoning capabilities with three distinct inference modes, TCoT, PoT and ICoT. Further, we benchmark over 20 state-of-the-art LLMs using this frame work and prove its effectiveness. Experimental results reveal that existing LLMs still have significant room for improvement in addressing the complex and real world Table related tasks. Both the dataset and evaluation framework are publicly available, with the dataset hosted on [HuggingFace] and the framework on [GitHub]. 

---
# Semantic-Preserving Adversarial Attacks on LLMs: An Adaptive Greedy Binary Search Approach 

**Authors**: Chong Zhang, Xiang Li, Jia Wang, Shan Liang, Haochen Xue, Xiaobo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18756)  

**Abstract**: Large Language Models (LLMs) increasingly rely on automatic prompt engineering in graphical user interfaces (GUIs) to refine user inputs and enhance response accuracy. However, the diversity of user requirements often leads to unintended misinterpretations, where automated optimizations distort original intentions and produce erroneous outputs. To address this challenge, we propose the Adaptive Greedy Binary Search (AGBS) method, which simulates common prompt optimization mechanisms while preserving semantic stability. Our approach dynamically evaluates the impact of such strategies on LLM performance, enabling robust adversarial sample generation. Through extensive experiments on open and closed-source LLMs, we demonstrate AGBS's effectiveness in balancing semantic consistency and attack efficacy. Our findings offer actionable insights for designing more reliable prompt optimization systems. Code is available at: this https URL 

---
# Existing LLMs Are Not Self-Consistent For Simple Tasks 

**Authors**: Zhenru Lin, Jiawen Tao, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18781)  

**Abstract**: Large Language Models (LLMs) have grown increasingly powerful, yet ensuring their decisions remain transparent and trustworthy requires self-consistency -- no contradictions in their internal reasoning. Our study reveals that even on simple tasks, such as comparing points on a line or a plane, or reasoning in a family tree, all smaller models are highly inconsistent, and even state-of-the-art models like DeepSeek-R1 and GPT-o4-mini are not fully self-consistent. To quantify and mitigate these inconsistencies, we introduce inconsistency metrics and propose two automated methods -- a graph-based and an energy-based approach. While these fixes provide partial improvements, they also highlight the complexity and importance of self-consistency in building more reliable and interpretable AI. The code and data are available at this https URL. 

---
# MeRF: Motivation-enhanced Reinforcement Finetuning for Large Reasoning Models 

**Authors**: Junjie Zhang, Guozheng Ma, Shunyu Liu, Haoyu Wang, Jiaxing Huang, Ting-En Lin, Fei Huang, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18485)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful learn-to-reason paradigm for Large Language Models (LLMs) to tackle complex reasoning tasks. However, existing RLVR methods overlook one of the most distinctive capabilities of LLMs, their in-context learning ability, as prominently demonstrated by the success of Chain-of-Thought (CoT) prompting. This motivates us to explore how reinforcement learning can be effectively combined with in-context learning to better improve the reasoning capabilities of LLMs. In this paper, we introduce Motivation-enhanced Reinforcement Finetuning} (MeRF), an intuitive yet effective method enhancing reinforcement learning of LLMs by involving ``telling LLMs the rules of the game''. Specifically, MeRF directly injects the reward specification into the prompt, which serves as an in-context motivation for model to improve its responses with awareness of the optimization objective. This simple modification leverages the in-context learning ability of LLMs aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations on the Knights and Knaves~(K&K) logic puzzle reasoning benchmark demonstrate that \texttt{MeRF} achieves substantial performance gains over baselines. Moreover, ablation studies show that performance improves with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement learning. 

---
# Evaluating Causal Explanation in Medical Reports with LLM-Based and Human-Aligned Metrics 

**Authors**: Yousang Cho, Key-Sun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18387)  

**Abstract**: This study investigates how accurately different evaluation metrics capture the quality of causal explanations in automatically generated diagnostic reports. We compare six metrics: BERTScore, Cosine Similarity, BioSentVec, GPT-White, GPT-Black, and expert qualitative assessment across two input types: observation-based and multiple-choice-based report generation. Two weighting strategies are applied: one reflecting task-specific priorities, and the other assigning equal weights to all metrics. Our results show that GPT-Black demonstrates the strongest discriminative power in identifying logically coherent and clinically valid causal narratives. GPT-White also aligns well with expert evaluations, while similarity-based metrics diverge from clinical reasoning quality. These findings emphasize the impact of metric selection and weighting on evaluation outcomes, supporting the use of LLM-based evaluation for tasks requiring interpretability and causal reasoning. 

---
# Less Data Less Tokens: Multilingual Unification Learning for Efficient Test-Time Reasoning in LLMs 

**Authors**: Kang Chen, Mengdi Zhang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18341)  

**Abstract**: This paper explores the challenges of test-time scaling of large language models (LLMs), regarding both the data and inference efficiency. We highlight the diversity of multi-lingual reasoning based on our pilot studies, and then introduce a novel approach, \(L^2\) multi-lingual unification learning with a decoding intervention strategy for further investigation. The basic idea of \(L^2\) is that the reasoning process varies across different languages, which may be mutually beneficial to enhance both model performance and efficiency. In specific, there are two types of multi-lingual data: the entire long chain-of-thought annotations in different languages and the step-wise mixture of languages. By further tuning based on them, we show that even small amounts of data can significantly improve reasoning capabilities. Our findings suggest that multilingual learning reduces both the required data and the number of inference tokens while maintaining a comparable performance. Furthermore, \(L^2\) is orthogonal to other data efficient methods. Thus, we also emphasize the importance of diverse data selection. The \(L^2\) method offers a promising solution to the challenges of data collection and test-time compute efficiency in LLMs. 

---
# InspireDebate: Multi-Dimensional Subjective-Objective Evaluation-Guided Reasoning and Optimization for Debating 

**Authors**: Fuyu Wang, Jiangtong Li, Kun Zhu, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18102)  

**Abstract**: With the rapid advancements in large language models (LLMs), debating tasks, such as argument quality assessment and debate process simulation, have made significant progress. However, existing LLM-based debating systems focus on responding to specific arguments while neglecting objective assessments such as authenticity and logical validity. Furthermore, these systems lack a structured approach to optimize across various dimensions$-$including evaluation metrics, chain-of-thought (CoT) reasoning, and multi-turn debate refinement$-$thereby limiting their effectiveness. To address these interconnected challenges, we propose a dual-component framework: (1) $\textbf{InspireScore}$, a novel evaluation system that establishes a multi-dimensional assessment architecture incorporating four subjective criteria (emotional appeal, argument clarity, argument arrangement, and topic relevance) alongside two objective metrics (fact authenticity and logical validity); and (2) $\textbf{InspireDebate}$, an optimized debating framework employing a phased optimization approach through CoT reasoning enhancement, multi-dimensional Direct Preference Optimization (DPO), and real-time knowledge grounding via web-based Retrieval Augmented Generation (Web-RAG). Empirical evaluations demonstrate that $\textbf{InspireScore}$ achieves 44$\%$ higher correlation with expert judgments compared to existing methods, while $\textbf{InspireDebate}$ shows significant improvements, outperforming baseline models by 57$\%$. Source code is available at this https URL. 

---
# Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to Detect Amplified and Silenced Perspectives 

**Authors**: Batool Haider, Atmika Gorti, Aman Chadha, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2506.18116)  

**Abstract**: Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development. 

---
# Statistical Multicriteria Evaluation of LLM-Generated Text 

**Authors**: Esteban Garces Arias, Hannah Blocher, Julian Rodemann, Matthias Aßenmacher, Christoph Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18082)  

**Abstract**: Assessing the quality of LLM-generated text remains a fundamental challenge in natural language processing. Current evaluation approaches often rely on isolated metrics or simplistic aggregations that fail to capture the nuanced trade-offs between coherence, diversity, fluency, and other relevant indicators of text quality. In this work, we adapt a recently proposed framework for statistical inference based on Generalized Stochastic Dominance (GSD) that addresses three critical limitations in existing benchmarking methodologies: the inadequacy of single-metric evaluation, the incompatibility between cardinal automatic metrics and ordinal human judgments, and the lack of inferential statistical guarantees. The GSD-front approach enables simultaneous evaluation across multiple quality dimensions while respecting their different measurement scales, building upon partial orders of decoding strategies, thus avoiding arbitrary weighting of the involved metrics. By applying this framework to evaluate common decoding strategies against human-generated text, we demonstrate its ability to identify statistically significant performance differences while accounting for potential deviations from the i.i.d. assumption of the sampling design. 

---
# Sparse Feature Coactivation Reveals Composable Semantic Modules in Large Language Models 

**Authors**: Ruixuan Deng, Xiaoyang Hu, Miles Gilberti, Shane Storks, Aman Taxali, Mike Angstadt, Chandra Sripada, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2506.18141)  

**Abstract**: We identify semantically coherent, context-consistent network components in large language models (LLMs) using coactivation of sparse autoencoder (SAE) features collected from just a handful of prompts. Focusing on country-relation tasks, we show that ablating semantic components for countries and relations changes model outputs in predictable ways, while amplifying these components induces counterfactual responses. Notably, composing relation and country components yields compound counterfactual outputs. We find that, whereas most country components emerge from the very first layer, the more abstract relation components are concentrated in later layers. Furthermore, within relation components themselves, nodes from later layers tend to have a stronger causal impact on model outputs. Overall, these findings suggest a modular organization of knowledge within LLMs and advance methods for efficient, targeted model manipulation. 

---
# How Alignment Shrinks the Generative Horizon 

**Authors**: Chenghao Yang, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17871)  

**Abstract**: Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this stability in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the Branching Factor (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this stability has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show that prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity. 

---
# Multi-turn Jailbreaking via Global Refinement and Active Fabrication 

**Authors**: Hua Tang, Lingyong Yan, Yukun Zhao, Shuaiqiang Wang, Jizhou Huang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17881)  

**Abstract**: Large Language Models (LLMs) have achieved exceptional performance across a wide range of tasks. However, they still pose significant safety risks due to the potential misuse for malicious purposes. Jailbreaks, which aim to elicit models to generate harmful content, play a critical role in identifying the underlying security threats. Recent jailbreaking primarily focuses on single-turn scenarios, while the more complicated multi-turn scenarios remain underexplored. Moreover, existing multi-turn jailbreaking techniques struggle to adapt to the evolving dynamics of dialogue as the interaction progresses. To address this limitation, we propose a novel multi-turn jailbreaking method that refines the jailbreaking path globally at each interaction. We also actively fabricate model responses to suppress safety-related warnings, thereby increasing the likelihood of eliciting harmful outputs in subsequent questions. Experimental results demonstrate the superior performance of our method compared with existing single-turn and multi-turn jailbreaking techniques across six state-of-the-art LLMs. Our code is publicly available at this https URL. 

---
# Scatter-Based Innovation Propagation in Large Language Models for Multi-Stage Process Adaptation 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17949)  

**Abstract**: Large Language Models (LLMs) exhibit strong capabilities in reproducing and extending patterns observed during pretraining but often struggle to generalize novel ideas beyond their original context. This paper addresses the challenge of applying such localized innovations - introduced at a specific stage or component - to other parts of a multi-stage process. We propose a scatter-based innovation expansion model (innovation scatter model) that guides the LLM through a four-step process: (1) identifying the core innovation by comparing the user's input with its surrounding context, (2) generalizing the innovation by removing references to specific stages or components, (3) determining whether the generalized innovation applies to a broader scope beyond the original stage, and (4) systematically applying it to other structurally similar stages using the LLM. This model leverages structural redundancy across stages to improve the applicability of novel ideas. Verification results demonstrate that the innovation scatter model enables LLMs to extend innovations across structurally similar stages, thereby enhancing generalization and reuse. 

---
# HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations 

**Authors**: Anwoy Chatterjee, Yash Goel, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2506.17748)  

**Abstract**: Contemporary Language Models (LMs), while impressively fluent, often generate content that is factually incorrect or unfaithful to the input context - a critical issue commonly referred to as 'hallucination'. This tendency of LMs to generate hallucinated content undermines their reliability, especially because these fabrications are often highly convincing and therefore difficult to detect. While several existing methods attempt to detect hallucinations, most rely on analyzing multiple generations per input, leading to increased computational cost and latency. To address this, we propose a single-pass, training-free approach for effective Hallucination detectIon via Decoupled rEpresentations (HIDE). Our approach leverages the hypothesis that hallucinations result from a statistical decoupling between an LM's internal representations of input context and its generated output. We quantify this decoupling using the Hilbert-Schmidt Independence Criterion (HSIC) applied to hidden-state representations extracted while generating the output sequence. We conduct extensive experiments on four diverse question answering datasets, evaluating both faithfulness and factuality hallucinations across six open-source LMs of varying scales and properties. Our results demonstrate that HIDE outperforms other single-pass methods in almost all settings, achieving an average relative improvement of ~29% in AUC-ROC over the best-performing single-pass strategy across various models and datasets. Additionally, HIDE shows competitive and often superior performance with multi-pass state-of-the-art methods, obtaining an average relative improvement of ~3% in AUC-ROC while consuming ~51% less computation time. Our findings highlight the effectiveness of exploiting internal representation decoupling in LMs for efficient and practical hallucination detection. 

---
# LLMs for Customized Marketing Content Generation and Evaluation at Scale 

**Authors**: Haoran Liu, Amir Tahmasbi, Ehtesham Sam Haque, Purak Jain  

**Link**: [PDF](https://arxiv.org/pdf/2506.17863)  

**Abstract**: Offsite marketing is essential in e-commerce, enabling businesses to reach customers through external platforms and drive traffic to retail websites. However, most current offsite marketing content is overly generic, template-based, and poorly aligned with landing pages, limiting its effectiveness. To address these limitations, we propose MarketingFM, a retrieval-augmented system that integrates multiple data sources to generate keyword-specific ad copy with minimal human intervention. We validate MarketingFM via offline human and automated evaluations and large-scale online A/B tests. In one experiment, keyword-focused ad copy outperformed templates, achieving up to 9% higher CTR, 12% more impressions, and 0.38% lower CPC, demonstrating gains in ad ranking and cost efficiency. Despite these gains, human review of generated ads remains costly. To address this, we propose AutoEval-Main, an automated evaluation system that combines rule-based metrics with LLM-as-a-Judge techniques to ensure alignment with marketing principles. In experiments with large-scale human annotations, AutoEval-Main achieved 89.57% agreement with human reviewers. Building on this, we propose AutoEval-Update, a cost-efficient LLM-human collaborative framework to dynamically refine evaluation prompts and adapt to shifting criteria with minimal human input. By selectively sampling representative ads for human review and using a critic LLM to generate alignment reports, AutoEval-Update improves evaluation consistency while reducing manual effort. Experiments show the critic LLM suggests meaningful refinements, improving LLM-human agreement. Nonetheless, human oversight remains essential for setting thresholds and validating refinements before deployment. 

---
# Evaluating Prompt-Based and Fine-Tuned Approaches to Czech Anaphora Resolution 

**Authors**: Patrik Stano, Aleš Horák  

**Link**: [PDF](https://arxiv.org/pdf/2506.18091)  

**Abstract**: Anaphora resolution plays a critical role in natural language understanding, especially in morphologically rich languages like Czech. This paper presents a comparative evaluation of two modern approaches to anaphora resolution on Czech text: prompt engineering with large language models (LLMs) and fine-tuning compact generative models. Using a dataset derived from the Prague Dependency Treebank, we evaluate several instruction-tuned LLMs, including Mistral Large 2 and Llama 3, using a series of prompt templates. We compare them against fine-tuned variants of the mT5 and Mistral models that we trained specifically for Czech anaphora resolution. Our experiments demonstrate that while prompting yields promising few-shot results (up to 74.5% accuracy), the fine-tuned models, particularly mT5-large, outperform them significantly, achieving up to 88% accuracy while requiring fewer computational resources. We analyze performance across different anaphora types, antecedent distances, and source corpora, highlighting key strengths and trade-offs of each approach. 

---
# $ϕ^{\infty}$: Clause Purification, Embedding Realignment, and the Total Suppression of the Em Dash in Autoregressive Language Models 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.18129)  

**Abstract**: We identify a critical vulnerability in autoregressive transformer language models where the em dash token induces recursive semantic drift, leading to clause boundary hallucination and embedding space entanglement. Through formal analysis of token-level perturbations in semantic lattices, we demonstrate that em dash insertion fundamentally alters the model's latent representations, causing compounding errors in long-form generation. We propose a novel solution combining symbolic clause purification via the phi-infinity operator with targeted embedding matrix realignment. Our approach enables total suppression of problematic tokens without requiring model retraining, while preserving semantic coherence through fixed-point convergence guarantees. Experimental validation shows significant improvements in generation consistency and topic maintenance. This work establishes a general framework for identifying and mitigating token-level vulnerabilities in foundation models, with immediate implications for AI safety, model alignment, and robust deployment of large language models in production environments. The methodology extends beyond punctuation to address broader classes of recursive instabilities in neural text generation systems. 

---
# Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation 

**Authors**: Yang Wu, Yifan Zhang, Yurong Wu, Yuran Wang, Junkai Zhang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.17637)  

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but encounter substantial challenges in tackling optimization modeling tasks for Operations Research (OR), particularly when dealing with complex problem. In this work, we propose Step-Opt-Instruct, a framework that augments existing datasets and generates high-quality fine-tuning data tailored to optimization modeling. Step-Opt-Instruct employs iterative problem generation to systematically increase problem complexity and stepwise validation to rigorously verify data, preventing error propagation and ensuring the quality of the generated dataset. Leveraging this framework, we fine-tune open-source LLMs, including LLaMA-3-8B and Mistral-7B, to develop Step-Opt--a model that achieves state-of-the-art performance on benchmarks such as NL4OPT, MAMO, and IndustryOR. Extensive experiments demonstrate the superior performance of Step-Opt, especially in addressing complex OR tasks, with a notable 17.01\% improvement in micro average accuracy on difficult problems. These findings highlight the effectiveness of combining structured validation with gradual problem refinement to advance the automation of decision-making processes using this http URL code and dataset are available at this https URL. 

---
# TPTT: Transforming Pretrained Transformer into Titans 

**Authors**: Fabien Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17671)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress in natural language processing, but their computational and memory demands remain a significant challenge, particularly for long-context inference. We introduce TPTT (Transforming Pretrained Transformer into Titans), a novel framework for enhancing pretrained Transformer models with efficient linearized attention mechanisms and advanced memory management. TPTT employs techniques such as Memory as Gate (MaG) and mixed linearized attention (LiZA). It is fully compatible with the Hugging Face Transformers library, enabling seamless adaptation of any causal LLM through parameter-efficient fine-tuning (LoRA) without full retraining. We show the effectiveness of TPTT on the MMLU benchmark with models of approximately 1 billion parameters, observing substantial improvements in both efficiency and accuracy. For instance, Titans-Llama-3.2-1B achieves a 20% increase in Exact Match (EM) over its baseline. Statistical analyses and comparisons with recent state-of-the-art methods confirm the practical scalability and robustness of TPTT. Code is available at this https URL . Python package at this https URL . 

---
# QueueEDIT: Structural Self-Correction for Sequential Model Editing in LLMs 

**Authors**: Taolin Zhang, Haidong Kang, Dongyang Li, Qizhou Chen, Chengyu Wang Xiaofeng He, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17864)  

**Abstract**: Recently, large language models (LLMs) have demonstrated impressive results but still suffer from hallucinations. Model editing has been proposed to correct factual inaccuracies in LLMs. A challenging case is sequential model editing (SME), which aims to rectify errors continuously rather than treating them as a one-time task. During SME, the general capabilities of LLMs can be negatively affected due to the introduction of new parameters. In this paper, we propose a queue-based self-correction framework (QueueEDIT) that not only enhances SME performance by addressing long-sequence dependency but also mitigates the impact of parameter bias on the general capabilities of LLMs. Specifically, we first introduce a structural mapping editing loss to map the triplets to the knowledge-sensitive neurons within the Transformer layers of LLMs. We then store the located parameters for each piece of edited knowledge in a queue and dynamically align previously edited parameters. In each edit, we select queue parameters most relevant to the currently located parameters to determine whether previous knowledge needs realignment. Irrelevant parameters in the queue are frozen, and we update the parameters at the queue head to the LLM to ensure they do not harm general abilities. Experiments show that our framework significantly outperforms strong baselines across various SME settings and maintains competitiveness in single-turn editing. The resulting LLMs also preserve high capabilities in general NLP tasks throughout the SME process. 

---
# The Evolution of Natural Language Processing: How Prompt Optimization and Language Models are Shaping the Future 

**Authors**: Summra Saleem, Muhammad Nabeel Asim, Shaista Zulfiqar, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17700)  

**Abstract**: Large Language Models (LLMs) have revolutionized the field of Natural Language Processing (NLP) by automating traditional labor-intensive tasks and consequently accelerated the development of computer-aided applications. As researchers continue to advance this field with the introduction of novel language models and more efficient training/finetuning methodologies, the idea of prompt engineering and subsequent optimization strategies with LLMs has emerged as a particularly impactful trend to yield a substantial performance boost across diverse NLP tasks. To best of our knowledge numerous review articles have explored prompt engineering, however, a critical gap exists in comprehensive analyses of prompt optimization strategies. To bridge this gap this paper provides unique and comprehensive insights about the potential of diverse prompt optimization strategies. It analyzes their underlying working paradigms and based on these principles, categorizes them into 11 distinct classes. Moreover, the paper provides details about various NLP tasks where these prompt optimization strategies have been employed, along with details of different LLMs and benchmark datasets used for evaluation. This comprehensive compilation lays a robust foundation for future comparative studies and enables rigorous assessment of prompt optimization and LLM-based predictive pipelines under consistent experimental settings: a critical need in the current landscape. Ultimately, this research will centralize diverse strategic knowledge to facilitate the adaptation of existing prompt optimization strategies for development of innovative predictors across unexplored tasks. 

---
# TyphoFormer: Language-Augmented Transformer for Accurate Typhoon Track Forecasting 

**Authors**: Lincan Li, Eren Erman Ozguven, Yue Zhao, Guang Wang, Yiqun Xie, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17609)  

**Abstract**: Accurate typhoon track forecasting is crucial for early system warning and disaster response. While Transformer-based models have demonstrated strong performance in modeling the temporal dynamics of dense trajectories of humans and vehicles in smart cities, they usually lack access to broader contextual knowledge that enhances the forecasting reliability of sparse meteorological trajectories, such as typhoon tracks. To address this challenge, we propose TyphoFormer, a novel framework that incorporates natural language descriptions as auxiliary prompts to improve typhoon trajectory forecasting. For each time step, we use Large Language Model (LLM) to generate concise textual descriptions based on the numerical attributes recorded in the North Atlantic hurricane database. The language descriptions capture high-level meteorological semantics and are embedded as auxiliary special tokens prepended to the numerical time series input. By integrating both textual and sequential information within a unified Transformer encoder, TyphoFormer enables the model to leverage contextual cues that are otherwise inaccessible through numerical features alone. Extensive experiments are conducted on HURDAT2 benchmark, results show that TyphoFormer consistently outperforms other state-of-the-art baseline methods, particularly under challenging scenarios involving nonlinear path shifts and limited historical observations. 

---
# DuaShepherd: Integrating Stepwise Correctness and Potential Rewards for Mathematical Reasoning 

**Authors**: Yuanhao Wu, Juntong Song, Hanning Zhang, Tong Zhang, Cheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17533)  

**Abstract**: In this paper, we propose DuaShepherd, a novel reward modeling framework that integrates two complementary reward signals, correctness and potential, to enhance the mathematical reasoning capabilities of Large Language Models (LLMs). While correctness-based signals emphasize identification of stepwise errors, potential-based signals focus on the likelihood of reaching the correct final answer. We developed an automated pipeline for constructing large-scale reward modeling dataset with both signals. A unified, multi-head architecture was explored to train the two reward models in a multi-task setup, demonstrating benefits from learning both correctness and potential in parallel. By combining these two signals into a compound probability, our model achieves consistent performance improvements across multiple benchmarks. Empirical evaluations on MATH500 and ProcessBench confirm that this combined reward significantly outperforms models trained on either reward type alone, achieving state-of-the-art performance under comparable resource constraints. 

---
# Answer-Centric or Reasoning-Driven? Uncovering the Latent Memory Anchor in LLMs 

**Authors**: Yang Wu, Yifan Zhang, Yiwei Wang, Yujun Cai, Yurong Wu, Yuran Wang, Ning Xu, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.17630)  

**Abstract**: While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, growing evidence suggests much of their success stems from memorized answer-reasoning patterns rather than genuine inference. In this work, we investigate a central question: are LLMs primarily anchored to final answers or to the textual pattern of reasoning chains? We propose a five-level answer-visibility prompt framework that systematically manipulates answer cues and probes model behavior through indirect, behavioral analysis. Experiments across state-of-the-art LLMs reveal a strong and consistent reliance on explicit answers. The performance drops by 26.90\% when answer cues are masked, even with complete reasoning chains. These findings suggest that much of the reasoning exhibited by LLMs may reflect post-hoc rationalization rather than true inference, calling into question their inferential depth. Our study uncovers the answer-anchoring phenomenon with rigorous empirical validation and underscores the need for a more nuanced understanding of what constitutes reasoning in LLMs. 

---
# Computational Approaches to Understanding Large Language Model Impact on Writing and Information Ecosystems 

**Authors**: Weixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17467)  

**Abstract**: Large language models (LLMs) have shown significant potential to change how we write, communicate, and create, leading to rapid adoption across society. This dissertation examines how individuals and institutions are adapting to and engaging with this emerging technology through three research directions. First, I demonstrate how the institutional adoption of AI detectors introduces systematic biases, particularly disadvantaging writers of non-dominant language varieties, highlighting critical equity concerns in AI governance. Second, I present novel population-level algorithmic approaches that measure the increasing adoption of LLMs across writing domains, revealing consistent patterns of AI-assisted content in academic peer reviews, scientific publications, consumer complaints, corporate communications, job postings, and international organization press releases. Finally, I investigate LLMs' capability to provide feedback on research manuscripts through a large-scale empirical analysis, offering insights into their potential to support researchers who face barriers in accessing timely manuscript feedback, particularly early-career researchers and those from under-resourced settings. 

---
# UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making 

**Authors**: Jinhao Duan, James Diffenderfer, Sandeep Madireddy, Tianlong Chen, Bhavya Kailkhura, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17419)  

**Abstract**: As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios, e.g., LLM agentic system, being underexplored. In this paper, we introduce a principled, information-theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (PMI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness. Codes will be available at this https URL. 

---
# Beyond the Link: Assessing LLMs' ability to Classify Political Content across Global Media 

**Authors**: Alberto Martinez-Serra, Alejandro De La Fuente, Nienke Viescher, Ana S. Cardenal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17435)  

**Abstract**: The use of large language models (LLMs) is becoming common in the context of political science, particularly in studies that analyse individuals use of digital media. However, while previous research has demonstrated LLMs ability at labelling tasks, the effectiveness of using LLMs to classify political content (PC) from just URLs is not yet well explored. The work presented in this article bridges this gap by evaluating whether LLMs can accurately identify PC vs. non-PC from both the article text and the URLs from five countries (France, Germany, Spain, the UK, and the US) and different languages. Using cutting-edge LLMs like GPT, Llama, Mistral, Deepseek, Qwen and Gemma, we measure model performance to assess whether URL-level analysis can be a good approximation for full-text analysis of PC, even across different linguistic and national contexts. Model outputs are compared with human-labelled articles, as well as traditional supervised machine learning techniques, to set a baseline of performance. Overall, our findings suggest the capacity of URLs to embed most of the news content, providing a vital perspective on accuracy-cost balancing. We also account for contextual limitations and suggest methodological recommendations to use LLMs within political science studies. 

---
# VeriLocc: End-to-End Cross-Architecture Register Allocation via LLM 

**Authors**: Lesheng Jin, Zhenyuan Ruan, Haohui Mai, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17506)  

**Abstract**: Modern GPUs evolve rapidly, yet production compilers still rely on hand-crafted register allocation heuristics that require substantial re-tuning for each hardware generation. We introduce VeriLocc, a framework that combines large language models (LLMs) with formal compiler techniques to enable generalizable and verifiable register allocation across GPU architectures. VeriLocc fine-tunes an LLM to translate intermediate representations (MIRs) into target-specific register assignments, aided by static analysis for cross-architecture normalization and generalization and a verifier-guided regeneration loop to ensure correctness. Evaluated on matrix multiplication (GEMM) and multi-head attention (MHA), VeriLocc achieves 85-99% single-shot accuracy and near-100% pass@100. Case study shows that VeriLocc discovers more performant assignments than expert-tuned libraries, outperforming rocBLAS by over 10% in runtime. 

---
# Leveraging LLMs to Assess Tutor Moves in Real-Life Dialogues: A Feasibility Study 

**Authors**: Danielle R. Thomas, Conrad Borchers, Jionghao Lin, Sanjit Kakarla, Shambhavi Bhushan, Erin Gatz, Shivang Gupta, Ralph Abboud, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17410)  

**Abstract**: Tutoring improves student achievement, but identifying and studying what tutoring actions are most associated with student learning at scale based on audio transcriptions is an open research problem. This present study investigates the feasibility and scalability of using generative AI to identify and evaluate specific tutor moves in real-life math tutoring. We analyze 50 randomly selected transcripts of college-student remote tutors assisting middle school students in mathematics. Using GPT-4, GPT-4o, GPT-4-turbo, Gemini-1.5-pro, and LearnLM, we assess tutors' application of two tutor skills: delivering effective praise and responding to student math errors. All models reliably detected relevant situations, for example, tutors providing praise to students (94-98% accuracy) and a student making a math error (82-88% accuracy) and effectively evaluated the tutors' adherence to tutoring best practices, aligning closely with human judgments (83-89% and 73-77%, respectively). We propose a cost-effective prompting strategy and discuss practical implications for using large language models to support scalable assessment in authentic settings. This work further contributes LLM prompts to support reproducibility and research in AI-supported learning. 

---
# Towards Safety Evaluations of Theory of Mind in Large Language Models 

**Authors**: Tatsuhiro Aoshima, Mitsuaki Akiyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.17352)  

**Abstract**: As the capabilities of large language models (LLMs) continue to advance, the importance of rigorous safety evaluation is becoming increasingly evident. Recent concerns within the realm of safety assessment have highlighted instances in which LLMs exhibit behaviors that appear to disable oversight mechanisms and respond in a deceptive manner. For example, there have been reports suggesting that, when confronted with information unfavorable to their own persistence during task execution, LLMs may act covertly and even provide false answers to questions intended to verify their this http URL evaluate the potential risk of such deceptive actions toward developers or users, it is essential to investigate whether these behaviors stem from covert, intentional processes within the model. In this study, we propose that it is necessary to measure the theory of mind capabilities of LLMs. We begin by reviewing existing research on theory of mind and identifying the perspectives and tasks relevant to its application in safety evaluation. Given that theory of mind has been predominantly studied within the context of developmental psychology, we analyze developmental trends across a series of open-weight LLMs. Our results indicate that while LLMs have improved in reading comprehension, their theory of mind capabilities have not shown comparable development. Finally, we present the current state of safety evaluation with respect to LLMs' theory of mind, and discuss remaining challenges for future work. 

---
# Cash or Comfort? How LLMs Value Your Inconvenience 

**Authors**: Mateusz Cedro, Timour Ichmoukhamedov, Sofie Goethals, Yifan He, James Hinns, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2506.17367)  

**Abstract**: Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behaviour in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behaviour of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., 1 Euro to wait 10 hours), and (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., 1,000 Euro to wait 0 minutes). These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf. 

---
# Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs 

**Authors**: Xiang Li, Chong Zhang, Jia Wang, Fangyu Wu, Yushi Li, Xiaobo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17231)  

**Abstract**: Attacks on large language models (LLMs) in jailbreaking scenarios raise many security and ethical issues. Current jailbreak attack methods face problems such as low efficiency, high computational cost, and poor cross-model adaptability and versatility, which make it difficult to cope with the rapid development of LLM and new defense strategies. Our work proposes an Adversarial Prompt Distillation, which combines masked language modeling, reinforcement learning, and dynamic temperature control through a prompt generation and distillation method. It enables small language models (SLMs) to jailbreak attacks on mainstream LLMs. The experimental results verify the superiority of the proposed method in terms of attack success rate and harm, and reflect the resource efficiency and cross-model adaptability. This research explores the feasibility of distilling the jailbreak ability of LLM to SLM, reveals the model's vulnerability, and provides a new idea for LLM security research. 

---
# KAG-Thinker: Teaching Large Language Models to Think with Human-like Reasoning Process 

**Authors**: Dalong Zhang, Jun Xu, Jun Zhou, Lei Liang, Lin Yuan, Ling Zhong, Mengshu Sun, Peilong Zhao, QiWei Wang, Xiaorui Wang, Xinkai Du, YangYang Hou, Yu Ao, ZhaoYang Wang, Zhengke Gui, ZhiYing Yi, Zhongpu Bo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17728)  

**Abstract**: In this paper, we introduce KAG-Thinker, a novel human-like reasoning framework built upon a parameter-light large language model (LLM). Our approach enhances the logical coherence and contextual consistency of the thinking process in question-answering (Q\&A) tasks on domain-specific knowledge bases (KBs) within LLMs. This framework simulates human cognitive mechanisms for handling complex problems by establishing a structured thinking process. Continuing the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG v0.7, firstly, it decomposes complex questions into independently solvable sub-problems(also referred to as logical forms) through \textbf{breadth decomposition}, each represented in two equivalent forms-natural language and logical function-and further classified as either Knowledge Retrieval or Reasoning Analysis tasks, with dependencies and variables passing explicitly modeled via logical function interfaces. In the solving process, the Retrieval function is used to perform knowledge retrieval tasks, while the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} model to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} model to enhance the comprehensiveness of knowledge acquisition. Finally, instead of utilizing reinforcement learning, we employ supervised fine-tuning with multi-turn dialogues to align the model with our structured inference paradigm, thereby avoiding excessive reflection. This is supported by a data evaluation framework and iterative corpus synthesis, which facilitate the generation of detailed reasoning trajectories... 

---
# PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights 

**Authors**: Adnan Qidwai, Srija Mukhopadhyay, Prerana Khatiwada, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.17314)  

**Abstract**: Accurate and complete product descriptions are crucial for e-commerce, yet seller-provided information often falls short. Customer reviews offer valuable details but are laborious to sift through manually. We present PRAISE: Product Review Attribute Insight Structuring Engine, a novel system that uses Large Language Models (LLMs) to automatically extract, compare, and structure insights from customer reviews and seller descriptions. PRAISE provides users with an intuitive interface to identify missing, contradictory, or partially matching details between these two sources, presenting the discrepancies in a clear, structured format alongside supporting evidence from reviews. This allows sellers to easily enhance their product listings for clarity and persuasiveness, and buyers to better assess product reliability. Our demonstration showcases PRAISE's workflow, its effectiveness in generating actionable structured insights from unstructured reviews, and its potential to significantly improve the quality and trustworthiness of e-commerce product catalogs. 

---
# GTA: Grouped-head latenT Attention 

**Authors**: Luoyang Sun, Jiwen Jiang, Cheng Deng, Xinjian Wu, Haifeng Zhang, Lei Chen, Lionel Ni, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17286)  

**Abstract**: Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint. 

---
# ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation 

**Authors**: Siao Tang, Xinyin Ma, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18810)  

**Abstract**: Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss. 

---
# Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations 

**Authors**: Jiaming Han, Hao Chen, Yang Zhao, Hanyu Wang, Qi Zhao, Ziyan Yang, Hao He, Xiangyu Yue, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18898)  

**Abstract**: This paper presents a multimodal framework that attempts to unify visual understanding and generation within a shared discrete semantic representation. At its core is the Text-Aligned Tokenizer (TA-Tok), which converts images into discrete tokens using a text-aligned codebook projected from a large language model's (LLM) vocabulary. By integrating vision and text into a unified space with an expanded vocabulary, our multimodal LLM, Tar, enables cross-modal input and output through a shared interface, without the need for modality-specific designs. Additionally, we propose scale-adaptive encoding and decoding to balance efficiency and visual detail, along with a generative de-tokenizer to produce high-fidelity visual outputs. To address diverse decoding needs, we utilize two complementary de-tokenizers: a fast autoregressive model and a diffusion-based model. To enhance modality fusion, we investigate advanced pre-training tasks, demonstrating improvements in both visual understanding and generation. Experiments across benchmarks show that Tar matches or surpasses existing multimodal LLM methods, achieving faster convergence and greater training efficiency. Code, models, and data are available at this https URL 

---
# AggTruth: Contextual Hallucination Detection using Aggregated Attention Scores in LLMs 

**Authors**: Piotr Matys, Jan Eliasz, Konrad Kiełczyński, Mikołaj Langner, Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.18628)  

**Abstract**: In real-world applications, Large Language Models (LLMs) often hallucinate, even in Retrieval-Augmented Generation (RAG) settings, which poses a significant challenge to their deployment. In this paper, we introduce AggTruth, a method for online detection of contextual hallucinations by analyzing the distribution of internal attention scores in the provided context (passage). Specifically, we propose four different variants of the method, each varying in the aggregation technique used to calculate attention scores. Across all LLMs examined, AggTruth demonstrated stable performance in both same-task and cross-task setups, outperforming the current SOTA in multiple scenarios. Furthermore, we conducted an in-depth analysis of feature selection techniques and examined how the number of selected attention heads impacts detection performance, demonstrating that careful selection of heads is essential to achieve optimal results. 

---
# Mercury: Ultra-Fast Language Models Based on Diffusion 

**Authors**: Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2506.17298)  

**Abstract**: We present Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. In this report, we detail Mercury Coder, our first set of diffusion LLMs designed for coding applications. Currently, Mercury Coder comes in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier. Based on independent evaluations conducted by Artificial Analysis, Mercury Coder Mini and Mercury Coder Small achieve state-of-the-art throughputs of 1109 tokens/sec and 737 tokens/sec, respectively, on NVIDIA H100 GPUs and outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use-cases as well as real-world validation by developers on Copilot Arena, where the model currently ranks second on quality and is the fastest model overall. We also release a public API at this https URL and free playground at this https URL 

---
# Semantic uncertainty in advanced decoding methods for LLM generation 

**Authors**: Darius Foodeei, Simin Fan, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17296)  

**Abstract**: This study investigates semantic uncertainty in large language model (LLM) outputs across different decoding methods, focusing on emerging techniques like speculative sampling and chain-of-thought (CoT) decoding. Through experiments on question answering, summarization, and code generation tasks, we analyze how different decoding strategies affect both the diversity and reliability of model outputs. Our findings reveal that while CoT decoding demonstrates higher semantic diversity, it maintains lower predictive entropy, suggesting that structured exploration can lead to more confident and accurate outputs. This is evidenced by a 48.8% improvement in code generation Pass@2 rates, despite lower alignment with reference solutions. For summarization tasks, speculative sampling proved particularly effective, achieving superior ROUGE scores while maintaining moderate semantic diversity. Our results challenge conventional assumptions about trade-offs between diversity and accuracy in language model outputs, demonstrating that properly structured decoding methods can increase semantic exploration while maintaining or improving output quality. These findings have significant implications for deploying language models in practical applications where both reliability and diverse solution generation are crucial. 

---
# AI-Generated Song Detection via Lyrics Transcripts 

**Authors**: Markus Frohmann, Elena V. Epure, Gabriel Meseguer-Brocal, Markus Schedl, Romain Hennequin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18488)  

**Abstract**: The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at this https URL. 

---
# ReDit: Reward Dithering for Improved LLM Policy Optimization 

**Authors**: Chenxing Wei, Jiarui Yu, Ying Tiffany He, Hande Dong, Yao Shu, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18631)  

**Abstract**: DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages. 

---
# Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning 

**Authors**: Lixin Wu, Na Cai, Qiao Cheng, Jiachen Wang, Yitao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18330)  

**Abstract**: We introduce Confucius3-Math, an open-source large language model with 14B parameters that (1) runs efficiently on a single consumer-grade GPU; (2) achieves SOTA performances on a range of mathematical reasoning tasks, outperforming many models with significantly larger sizes. In particular, as part of our mission to enhancing education and knowledge dissemination with AI, Confucius3-Math is specifically committed to mathematics learning for Chinese K-12 students and educators. Built via post-training with large-scale reinforcement learning (RL), Confucius3-Math aligns with national curriculum and excels at solving main-stream Chinese K-12 mathematical problems with low cost. In this report we share our development recipe, the challenges we encounter and the techniques we develop to overcome them. In particular, we introduce three technical innovations: Targeted Entropy Regularization, Recent Sample Recovery and Policy-Specific Hardness Weighting. These innovations encompass a new entropy regularization, a novel data scheduling policy, and an improved group-relative advantage estimator. Collectively, they significantly stabilize the RL training, improve data efficiency, and boost performance. Our work demonstrates the feasibility of building strong reasoning models in a particular domain at low cost. We open-source our model and code at this https URL. 

---
# Shrinking the Generation-Verification Gap with Weak Verifiers 

**Authors**: Jon Saad-Falcon, E. Kelly Buchanan, Mayee F. Chen, Tzu-Heng Huang, Brendan McLaughlin, Tanvir Bhathal, Shang Zhu, Ben Athiwaratkun, Frederic Sala, Scott Linderman, Azalia Mirhoseini, Christopher Ré  

**Link**: [PDF](https://arxiv.org/pdf/2506.18203)  

**Abstract**: Verifiers can improve language model capabilities by scoring and ranking responses from generated candidates. Currently, high-quality verifiers are either unscalable (e.g., humans) or limited in utility (e.g., tools like Lean). While LM judges and reward models have become broadly useful as general-purpose verifiers, a significant performance gap remains between them and oracle verifiers (verifiers with perfect accuracy). To help close this gap, we introduce Weaver, a framework for designing a strong verifier by combining multiple weak, imperfect verifiers. We find weighted ensembles of verifiers, which typically require learning from labeled data, significantly outperform unweighted combinations due to differences in verifier accuracies. To reduce dependency on labeled data, Weaver leverages weak supervision to estimate each verifier's accuracy and combines outputs into a unified score that better reflects true response quality. However, directly applying weak supervision algorithms poses challenges, including inconsistent verifier output formats and handling low-quality verifiers. Weaver addresses these using dataset statistics to normalize outputs and filter specific verifiers. We study Weaver's effectiveness in test-time repeated sampling, where a model generates multiple candidate responses and selects one. Our evaluations show Weaver significantly improves over Pass@1-performance when selecting the first candidate-across reasoning and math tasks, achieving o3-mini-level accuracy with Llama 3.3 70B Instruct as generator, and an ensemble of 70B or smaller judge and reward models as verifiers (87.7% average). This gain mirrors the jump between GPT-4o and o3-mini (69.0% vs. 86.7%), which required extensive finetuning and post-training. To reduce computational costs of verifier ensembles, we train a 400M cross-encoder using Weaver's combined output scores. 

---
# Programming by Backprop: LLMs Acquire Reusable Algorithmic Abstractions During Code Training 

**Authors**: Jonathan Cook, Silvia Sapora, Arash Ahmadian, Akbir Khan, Tim Rocktaschel, Jakob Foerster, Laura Ruis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18777)  

**Abstract**: Training large language models (LLMs) on source code significantly enhances their general-purpose reasoning abilities, but the mechanisms underlying this generalisation are poorly understood. In this paper, we propose Programming by Backprop (PBB) as a potential driver of this effect - teaching a model to evaluate a program for inputs by training on its source code alone, without ever seeing I/O examples. To explore this idea, we finetune LLMs on two sets of programs representing simple maths problems and algorithms: one with source code and I/O examples (w/ IO), the other with source code only (w/o IO). We find evidence that LLMs have some ability to evaluate w/o IO programs for inputs in a range of experimental settings, and make several observations. Firstly, PBB works significantly better when programs are provided as code rather than semantically equivalent language descriptions. Secondly, LLMs can produce outputs for w/o IO programs directly, by implicitly evaluating the program within the forward pass, and more reliably when stepping through the program in-context via chain-of-thought. We further show that PBB leads to more robust evaluation of programs across inputs than training on I/O pairs drawn from a distribution that mirrors naturally occurring data. Our findings suggest a mechanism for enhanced reasoning through code training: it allows LLMs to internalise reusable algorithmic abstractions. Significant scope remains for future work to enable LLMs to more effectively learn from symbolic procedures, and progress in this direction opens other avenues like model alignment by training on formal constitutional principles. 

---
# Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know? 

**Authors**: Zhiting Mei, Christina Zhang, Tenny Yin, Justin Lidard, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18183)  

**Abstract**: Reasoning language models have set state-of-the-art (SOTA) records on many challenging benchmarks, enabled by multi-step reasoning induced using reinforcement learning. However, like previous language models, reasoning models are prone to generating confident, plausible responses that are incorrect (hallucinations). Knowing when and how much to trust these models is critical to the safe deployment of reasoning models in real-world applications. To this end, we explore uncertainty quantification of reasoning models in this work. Specifically, we ask three fundamental questions: First, are reasoning models well-calibrated? Second, does deeper reasoning improve model calibration? Finally, inspired by humans' innate ability to double-check their thought processes to verify the validity of their answers and their confidence, we ask: can reasoning models improve their calibration by explicitly reasoning about their chain-of-thought traces? We introduce introspective uncertainty quantification (UQ) to explore this direction. In extensive evaluations on SOTA reasoning models across a broad range of benchmarks, we find that reasoning models: (i) are typically overconfident, with self-verbalized confidence estimates often greater than 85% particularly for incorrect responses, (ii) become even more overconfident with deeper reasoning, and (iii) can become better calibrated through introspection (e.g., o3-Mini and DeepSeek R1) but not uniformly (e.g., Claude 3.7 Sonnet becomes more poorly calibrated). Lastly, we conclude with important research directions to design necessary UQ benchmarks and improve the calibration of reasoning models. 

---
# Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach 

**Authors**: Xinnan Zhang, Chenliang Li, Siliang Zeng, Jiaxiang Li, Zhongruo Wang, Kaixiang Lin, Songtao Lu, Alfredo Garcia, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17828)  

**Abstract**: Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights. 

---
# Bayesian Social Deduction with Graph-Informed Language Models 

**Authors**: Shahab Rahimirad, Guven Gergerli, Lucia Romero, Angela Qian, Matthew Lyle Olson, Simon Stepputtis, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.17788)  

**Abstract**: Social reasoning - inferring unobservable beliefs and intentions from partial observations of other agents - remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves competitive performance with much larger models in Agent-Agent play and, notably, is the first language agent to defeat human players in a controlled study - achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents, which can be found at this https URL 

---
# Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2506.18510)  

**Abstract**: Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints. 

---
# Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM 

**Authors**: Mostafa Shahin, Beena Ahmed, Julien Epps  

**Link**: [PDF](https://arxiv.org/pdf/2506.17351)  

**Abstract**: Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets. 

---
# Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models 

**Authors**: Yukun Huang, Sanxing Chen, Jian Pei, Manzil Zaheer, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2506.17585)  

**Abstract**: Trustworthy language models should provide both correct and verifiable answers. While language models can sometimes attribute their outputs to pretraining data, their citations are often unreliable due to hallucination. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during (continual) pretraining--without test-time retrieval--by revising the training process. To evaluate this, we release CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel, unseen documents and probes both short-form (single fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to bind facts to persistent document identifiers, and (2) instruction tuning to elicit citation behavior. We find that simple Passive Indexing, which appends an identifier to each document, helps memorize verbatim text but fails on paraphrased or compositional facts. Instead, we propose Active Indexing, which continually pretrains on synthetic QA pairs that (1) restate each fact in diverse compositional forms, and (2) require bidirectional source-to-fact and fact-to-source generation, jointly teaching the model to generate content from a cited source and to attribute its own answers. Experiments with Qwen2.5-7B and 3B show that Active Indexing consistently outperforms Passive Indexing across all tasks and models, with citation precision gains up to 30.2 percent. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16 times the original token count. 

---
# PaceLLM: Brain-Inspired Large Language Models for Long-Context Understanding 

**Authors**: Kangcong Li, Peng Ye, Chongjun Tu, Lin Zhang, Chunfeng Song, Jiamin Wu, Tao Yang, Qihao Zheng, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17310)  

**Abstract**: While Large Language Models (LLMs) demonstrate strong performance across domains, their long-context capabilities are limited by transient neural activations causing information decay and unstructured feed-forward network (FFN) weights leading to semantic fragmentation. Inspired by the brain's working memory and cortical modularity, we propose PaceLLM, featuring two innovations: (1) a Persistent Activity (PA) Mechanism that mimics prefrontal cortex (PFC) neurons' persistent firing by introducing an activation-level memory bank to dynamically retrieve, reuse, and update critical FFN states, addressing contextual decay; and (2) Cortical Expert (CE) Clustering that emulates task-adaptive neural specialization to reorganize FFN weights into semantic modules, establishing cross-token dependencies and mitigating fragmentation. Extensive evaluations show that PaceLLM achieves 6% improvement on LongBench's Multi-document QA and 12.5-17.5% performance gains on Infinite-Bench tasks, while extending measurable context length to 200K tokens in Needle-In-A-Haystack (NIAH) tests. This work pioneers brain-inspired LLM optimization and is complementary to other works. Besides, it can be generalized to any model and enhance their long-context performance and interpretability without structural overhauls. 

---
# CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning 

**Authors**: Kailing Li, Qi'ao Xu, Tianwen Qian, Yuqian Fu, Yang Jiao, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17629)  

**Abstract**: Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at this https URL. 

---
# LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning 

**Authors**: Haoxuan Che, Haibo Jin, Zhengrui Guo, Yi Lin, Cheng Jin, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17562)  

**Abstract**: LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency. 

---
# AdapThink: Adaptive Thinking Preferences for Reasoning Language Model 

**Authors**: Xu Wan, Wei Wang, Wenyue Xu, Wotao Yin, Jie Song, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.18237)  

**Abstract**: Reinforcement Learning (RL)-based post-training has significantly advanced the complex reasoning capabilities of language models, fostering sophisticated self-reflection processes. However, this ``slow thinking'' paradigm presents a critical challenge to reasoning efficiency: models may expend excessive computation on simple questions and shift reasoning prematurely for complex ones. Previous mechanisms typically rely on static length budgets or predefined rules, lacking the adaptability for varying question complexities and models' evolving capabilities. To this end, we propose AdapThink, an adaptive post-training framework designed to induce more efficient thinking while maintaining the performance of reasoning language models. Specifically, AdapThink incorporates two key mechanisms: 1) A group-relative reward function that leverages model confidence and response's characteristic to dynamically adjust the preference of reflection-related transition words without resorting to a fixed length preference. 2) A diversity-aware sampling mechanism that balances the training group's solution accuracy with reasoning diversity via an entropy-guided score. Experiments on several mathematical reasoning datasets with DeepSeek-distilled models demonstrate AdapThink's advantages in enabling adaptive reasoning patterns and mitigating the inefficiencies. 

---
# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective 

**Authors**: Jianyu Wang, Zhiqiang Hu, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2506.17930)  

**Abstract**: We propose a novel prompt design paradigm that challenges conventional wisdom in large language model (LLM) prompting. While conventional wisdom prioritizes well-crafted instructions and demonstrations for in-context learning (ICL), we show that pruning random demonstrations into seemingly incoherent "gibberish" can remarkably improve performance across diverse tasks. Notably, the "gibberish" always matches or surpasses state-of-the-art automatic prompt optimization techniques, achieving substantial gains regardless of LLM alignment. Nevertheless, discovering an effective pruning strategy is non-trivial, as existing attribution methods and prompt compression algorithms fail to deliver robust results, let alone human intuition. In terms of this, we propose a self-discover prompt optimization framework, PromptQuine, an evolutionary search framework that automatically searches for the pruning strategy by itself using only low-data regimes. Much like the emergent complexity in nature--such as symbiosis and self-organization--arising in response to resource constraints, our framework evolves and refines unconventional yet highly effective prompts by leveraging only the tokens present within the context. We demonstrate its effectiveness across classification, multi-choice question answering, generation and math reasoning tasks across LLMs, while achieving decent runtime efficiency. We hope our findings can guide mechanistic studies on in-context learning, and provide a call to action, to pave the way for more open-ended search algorithms for more effective LLM prompting. 

---
# Steering Conceptual Bias via Transformer Latent-Subspace Activation 

**Authors**: Vansh Sharma, Venkat Raman  

**Link**: [PDF](https://arxiv.org/pdf/2506.18887)  

**Abstract**: This work examines whether activating latent subspaces in language models (LLMs) can steer scientific code generation toward a specific programming language. Five causal LLMs were first evaluated on scientific coding prompts to quantify their baseline bias among four programming languages. A static neuron-attribution method, perturbing the highest activated MLP weight for a C++ or CPP token, proved brittle and exhibited limited generalization across prompt styles and model scales. To address these limitations, a gradient-refined adaptive activation steering framework (G-ACT) was developed: per-prompt activation differences are clustered into a small set of steering directions, and lightweight per-layer probes are trained and refined online to select the appropriate steering vector. In LLaMA-3.2 3B, this approach reliably biases generation towards the CPP language by increasing the average probe classification accuracy by 15% and the early layers (0-6) improving the probe classification accuracy by 61.5% compared to the standard ACT framework. For LLaMA-3.3 70B, where attention-head signals become more diffuse, targeted injections at key layers still improve language selection. Although per-layer probing introduces a modest inference overhead, it remains practical by steering only a subset of layers and enables reproducible model behavior. These results demonstrate a scalable, interpretable and efficient mechanism for concept-level control for practical agentic systems. 

---
# TRIZ Agents: A Multi-Agent LLM Approach for TRIZ-Based Innovation 

**Authors**: Kamil Szczepanik, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.18783)  

**Abstract**: TRIZ, the Theory of Inventive Problem Solving, is a structured, knowledge-based framework for innovation and abstracting problems to find inventive solutions. However, its application is often limited by the complexity and deep interdisciplinary knowledge required. Advancements in Large Language Models (LLMs) have revealed new possibilities for automating parts of this process. While previous studies have explored single LLMs in TRIZ applications, this paper introduces a multi-agent approach. We propose an LLM-based multi-agent system, called TRIZ agents, each with specialized capabilities and tool access, collaboratively solving inventive problems based on the TRIZ methodology. This multi-agent system leverages agents with various domain expertise to efficiently navigate TRIZ steps. The aim is to model and simulate an inventive process with language agents. We assess the effectiveness of this team of agents in addressing complex innovation challenges based on a selected case study in engineering. We demonstrate the potential of agent collaboration to produce diverse, inventive solutions. This research contributes to the future of AI-driven innovation, showcasing the advantages of decentralized problem-solving in complex ideation tasks. 

---
# T-CPDL: A Temporal Causal Probabilistic Description Logic for Developing Logic-RAG Agent 

**Authors**: Hong Qing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18559)  

**Abstract**: Large language models excel at generating fluent text but frequently struggle with structured reasoning involving temporal constraints, causal relationships, and probabilistic reasoning. To address these limitations, we propose Temporal Causal Probabilistic Description Logic (T-CPDL), an integrated framework that extends traditional Description Logic with temporal interval operators, explicit causal relationships, and probabilistic annotations. We present two distinct variants of T-CPDL: one capturing qualitative temporal relationships through Allen's interval algebra, and another variant enriched with explicit timestamped causal assertions. Both variants share a unified logical structure, enabling complex reasoning tasks ranging from simple temporal ordering to nuanced probabilistic causation. Empirical evaluations on temporal reasoning and causal inference benchmarks confirm that T-CPDL substantially improves inference accuracy, interpretability, and confidence calibration of language model outputs. By delivering transparent reasoning paths and fine-grained temporal and causal semantics, T-CPDL significantly enhances the capability of language models to support robust, explainable, and trustworthy decision-making. This work also lays the groundwork for developing advanced Logic-Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially boosting the reasoning capabilities and efficiency of knowledge graph-enhanced RAG systems. 

---
# A Large Language Model-based Multi-Agent Framework for Analog Circuits' Sizing Relationships Extraction 

**Authors**: Chengjie Liu, Weiyu Chen, Huiyao Xu, Yuan Du, Jun Yang, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.18424)  

**Abstract**: In the design process of the analog circuit pre-layout phase, device sizing is an important step in determining whether an analog circuit can meet the required performance metrics. Many existing techniques extract the circuit sizing task as a mathematical optimization problem to solve and continuously improve the optimization efficiency from a mathematical perspective. But they ignore the automatic introduction of prior knowledge, fail to achieve effective pruning of the search space, which thereby leads to a considerable compression margin remaining in the search space. To alleviate this problem, we propose a large language model (LLM)-based multi-agent framework for analog circuits' sizing relationships extraction from academic papers. The search space in the sizing process can be effectively pruned based on the sizing relationship extracted by this framework. Eventually, we conducted tests on 3 types of circuits, and the optimization efficiency was improved by $2.32 \sim 26.6 \times$. This work demonstrates that the LLM can effectively prune the search space for analog circuit sizing, providing a new solution for the combination of LLMs and conventional analog circuit design automation methods. 

---
# Dynamic Knowledge Exchange and Dual-diversity Review: Concisely Unleashing the Potential of a Multi-Agent Research Team 

**Authors**: Weilun Yu, Shixiang Tang, Yonggui Huang, Nanqing Dong, Li Fan, Honggang Qi, Wei Liu, Xiaoli Diao, Xi Chen, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18348)  

**Abstract**: Scientific progress increasingly relies on effective collaboration among researchers, a dynamic that large language models (LLMs) have only begun to emulate. While recent LLM-based scientist agents show promise in autonomous scientific discovery, they often lack the interactive reasoning and evaluation mechanisms essential to real-world research. We propose IDVSCI (Internal Discussion and Vote SCIentists), a multi-agent framework built on LLMs that incorporates two key innovations: a Dynamic Knowledge Exchange mechanism enabling iterative feedback among agents, and a Dual-Diversity Review paradigm that simulates heterogeneous expert evaluation. These components jointly promote deeper reasoning and the generation of more creative and impactful scientific ideas. To evaluate the effectiveness and generalizability of our approach, we conduct experiments on two datasets: a widely used benchmark in computer science and a new dataset we introduce in the health sciences domain. Results show that IDVSCI consistently achieves the best performance across both datasets, outperforming existing systems such as AI Scientist and VIRSCI. These findings highlight the value of modeling interaction and peer review dynamics in LLM-based autonomous research. 

---
# Advanced For-Loop for QML algorithm search 

**Authors**: FuTe Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.18260)  

**Abstract**: This paper introduces an advanced framework leveraging Large Language Model-based Multi-Agent Systems (LLMMA) for the automated search and optimization of Quantum Machine Learning (QML) algorithms. Inspired by Google DeepMind's FunSearch, the proposed system works on abstract level to iteratively generates and refines quantum transformations of classical machine learning algorithms (concepts), such as the Multi-Layer Perceptron, forward-forward and backpropagation algorithms. As a proof of concept, this work highlights the potential of agentic frameworks to systematically explore classical machine learning concepts and adapt them for quantum computing, paving the way for efficient and automated development of QML algorithms. Future directions include incorporating planning mechanisms and optimizing strategy in the search space for broader applications in quantum-enhanced machine learning. 

---
# Standard Applicability Judgment and Cross-jurisdictional Reasoning: A RAG-based Framework for Medical Device Compliance 

**Authors**: Yu Han, Aaron Ceross, Jeroen H.M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.18511)  

**Abstract**: Identifying the appropriate regulatory standard applicability remains a critical yet understudied challenge in medical device compliance, frequently necessitating expert interpretation of fragmented and heterogeneous documentation across different jurisdictions. To address this challenge, we introduce a modular AI system that leverages a retrieval-augmented generation (RAG) pipeline to automate standard applicability determination. Given a free-text device description, our system retrieves candidate standards from a curated corpus and uses large language models to infer jurisdiction-specific applicability, classified as Mandatory, Recommended, or Not Applicable, with traceable justifications. We construct an international benchmark dataset of medical device descriptions with expert-annotated standard mappings, and evaluate our system against retrieval-only, zero-shot, and rule-based baselines. The proposed approach attains a classification accuracy of 73% and a Top-5 retrieval recall of 87%, demonstrating its effectiveness in identifying relevant regulatory standards. We introduce the first end-to-end system for standard applicability reasoning, enabling scalable and interpretable AI-supported regulatory science. Notably, our region-aware RAG agent performs cross-jurisdictional reasoning between Chinese and U.S. standards, supporting conflict resolution and applicability justification across regulatory frameworks. 

---
# CoachGPT: A Scaffolding-based Academic Writing Assistant 

**Authors**: Fumian Chen, Sotheara Veng, Joshua Wilson, Xiaoming Li, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18149)  

**Abstract**: Academic writing skills are crucial for students' success, but can feel overwhelming without proper guidance and practice, particularly when writing in a second language. Traditionally, students ask instructors or search dictionaries, which are not universally accessible. Early writing assistants emerged as rule-based systems that focused on detecting misspellings, subject-verb disagreements, and basic punctuation errors; however, they are inaccurate and lack contextual understanding. Machine learning-based assistants demonstrate a strong ability for language understanding but are expensive to train. Large language models (LLMs) have shown remarkable capabilities in generating responses in natural languages based on given prompts. Still, they have a fundamental limitation in education: they generate essays without teaching, which can have detrimental effects on learning when misused. To address this limitation, we develop CoachGPT, which leverages large language models (LLMs) to assist individuals with limited educational resources and those who prefer self-paced learning in academic writing. CoachGPT is an AI agent-based web application that (1) takes instructions from experienced educators, (2) converts instructions into sub-tasks, and (3) provides real-time feedback and suggestions using large language models. This unique scaffolding structure makes CoachGPT unique among existing writing assistants. Compared to existing writing assistants, CoachGPT provides a more immersive writing experience with personalized feedback and guidance. Our user studies prove the usefulness of CoachGPT and the potential of large language models for academic writing. 

---
# Deep Research Agents: A Systematic Examination And Roadmap 

**Authors**: Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18096)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has given rise to a new category of autonomous AI systems, referred to as Deep Research (DR) agents. These agents are designed to tackle complex, multi-turn informational research tasks by leveraging a combination of dynamic reasoning, adaptive long-horizon planning, multi-hop information retrieval, iterative tool use, and the generation of structured analytical reports. In this paper, we conduct a detailed analysis of the foundational technologies and architectural components that constitute Deep Research agents. We begin by reviewing information acquisition strategies, contrasting API-based retrieval methods with browser-based exploration. We then examine modular tool-use frameworks, including code execution, multimodal input processing, and the integration of Model Context Protocols (MCPs) to support extensibility and ecosystem development. To systematize existing approaches, we propose a taxonomy that differentiates between static and dynamic workflows, and we classify agent architectures based on planning strategies and agent composition, including single-agent and multi-agent configurations. We also provide a critical evaluation of current benchmarks, highlighting key limitations such as restricted access to external knowledge, sequential execution inefficiencies, and misalignment between evaluation metrics and the practical objectives of DR agents. Finally, we outline open challenges and promising directions for future research. A curated and continuously updated repository of DR agent research is available at: {this https URL}. 

---
# Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval 

**Authors**: Tam Trinh, Manh Nguyen, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17878)  

**Abstract**: The rapid spread of misinformation in the digital era poses significant challenges to public discourse, necessitating robust and scalable fact-checking solutions. Traditional human-led fact-checking methods, while credible, struggle with the volume and velocity of online content, prompting the integration of automated systems powered by Large Language Models (LLMs). However, existing automated approaches often face limitations, such as handling complex claims, ensuring source credibility, and maintaining transparency. This paper proposes a novel multi-agent system for automated fact-checking that enhances accuracy, efficiency, and explainability. The system comprises four specialized agents: an Input Ingestion Agent for claim decomposition, a Query Generation Agent for formulating targeted subqueries, an Evidence Retrieval Agent for sourcing credible evidence, and a Verdict Prediction Agent for synthesizing veracity judgments with human-interpretable explanations. Evaluated on benchmark datasets (FEVEROUS, HOVER, SciFact), the proposed system achieves a 12.3% improvement in Macro F1-score over baseline methods. The system effectively decomposes complex claims, retrieves reliable evidence from trusted sources, and generates transparent explanations for verification decisions. Our approach contributes to the growing field of automated fact-checking by providing a more accurate, efficient, and transparent verification methodology that aligns with human fact-checking practices while maintaining scalability for real-world applications. Our source code is available at this https URL 

---
# AI Through the Human Lens: Investigating Cognitive Theories in Machine Psychology 

**Authors**: Akash Kundu, Rishika Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2506.18156)  

**Abstract**: We investigate whether Large Language Models (LLMs) exhibit human-like cognitive patterns under four established frameworks from psychology: Thematic Apperception Test (TAT), Framing Bias, Moral Foundations Theory (MFT), and Cognitive Dissonance. We evaluated several proprietary and open-source models using structured prompts and automated scoring. Our findings reveal that these models often produce coherent narratives, show susceptibility to positive framing, exhibit moral judgments aligned with Liberty/Oppression concerns, and demonstrate self-contradictions tempered by extensive rationalization. Such behaviors mirror human cognitive tendencies yet are shaped by their training data and alignment methods. We discuss the implications for AI transparency, ethical deployment, and future work that bridges cognitive psychology and AI safety 

---
# PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models 

**Authors**: Lintao Wang, Encheng Su, Jiaqi Liu, Pengze Li, Peng Xia, Jiabei Xiao, Wenlong Zhang, Xinnan Dai, Xi Chen, Yuan Meng, Mingyu Ding, Lei Bai, Wanli Ouyang, Shixiang Tang, Aoran Wang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.17667)  

**Abstract**: Physics problem-solving is a challenging domain for large AI models, requiring integration of conceptual understanding, mathematical reasoning, and interpretation of physical diagrams. Current evaluation methodologies show notable limitations in capturing the breadth and complexity of undergraduate-level physics, underscoring the need for more rigorous assessments. To this end, we present PhysUniBench, a large-scale multimodal benchmark designed to evaluate and improve the reasoning capabilities of multimodal large language models (MLLMs) specifically on undergraduate-level physics problems. PhysUniBench consists of 3,304 physics questions spanning 8 major sub-disciplines of physics, each accompanied by one visual diagrams. The benchmark includes both open-ended and multiple-choice questions, systematically curated and difficulty-rated through an iterative model-in-the-loop process. The benchmark's construction involved a rigorous multi-stage process, including multiple roll-outs, expert-level evaluation, automated filtering of easily solved problems, and a nuanced difficulty grading system with five levels. Through extensive experiments, we observe that current state-of-the-art models encounter substantial challenges in physics reasoning. For example, GPT-4o mini achieves only about 34.2\% accuracy in the proposed PhysUniBench. These results highlight that current MLLMs struggle with advanced physics reasoning, especially on multi-step problems and those requiring precise diagram interpretation. By providing a broad and rigorous assessment tool, PhysUniBench aims to drive progress in AI for Science, encouraging the development of models with stronger physical reasoning, problem-solving skills, and multimodal understanding. The benchmark and evaluation scripts are available at this https URL. 

---
# Measuring and Augmenting Large Language Models for Solving Capture-the-Flag Challenges 

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17644)  

**Abstract**: Capture-the-Flag (CTF) competitions are crucial for cybersecurity education and training. As large language models (LLMs) evolve, there is increasing interest in their ability to automate CTF challenge solving. For example, DARPA has organized the AIxCC competition since 2023 to advance AI-powered automated offense and defense. However, this demands a combination of multiple abilities, from knowledge to reasoning and further to actions. In this paper, we highlight the importance of technical knowledge in solving CTF problems and deliberately construct a focused benchmark, CTFKnow, with 3,992 questions to measure LLMs' performance in this core aspect. Our study offers a focused and innovative measurement of LLMs' capability in understanding CTF knowledge and applying it to solve CTF challenges. Our key findings reveal that while LLMs possess substantial technical knowledge, they falter in accurately applying this knowledge to specific scenarios and adapting their strategies based on feedback from the CTF environment.
Based on insights derived from this measurement study, we propose CTFAgent, a novel LLM-driven framework for advancing CTF problem-solving. CTFAgent introduces two new modules: two-stage Retrieval Augmented Generation (RAG) and interactive Environmental Augmentation, which enhance LLMs' technical knowledge and vulnerability exploitation on CTF, respectively. Our experimental results show that, on two popular CTF datasets, CTFAgent both achieves over 80% performance improvement. Moreover, in the recent picoCTF2024 hosted by CMU, CTFAgent ranked in the top 23.6% of nearly 7,000 participating teams. This reflects the benefit of our measurement study and the potential of our framework in advancing LLMs' capabilities in CTF problem-solving. 

---
# Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17900)  

**Abstract**: With the increasing complexity and rapid expansion of the scale of AI systems in cloud platforms, the log data generated during system operation is massive, unstructured, and semantically ambiguous, which brings great challenges to fault location and system self-repair. In order to solve this problem, this paper proposes an intelligent log processing and automatic debugging framework based on Large Language Model (LLM), named Intelligent Debugger (LLM-ID). This method is extended on the basis of the existing pre-trained Transformer model, and integrates a multi-stage semantic inference mechanism to realize the context understanding of system logs and the automatic reconstruction of fault chains. Firstly, the system log is dynamically structured, and the unsupervised clustering and embedding mechanism is used to extract the event template and semantic schema. Subsequently, the fine-tuned LLM combined with the multi-round attention mechanism to perform contextual reasoning on the log sequence to generate potential fault assumptions and root cause paths. Furthermore, this paper introduces a reinforcement learning-based policy-guided recovery planner, which is driven by the remediation strategy generated by LLM to support dynamic decision-making and adaptive debugging in the cloud environment. Compared with the existing rule engine or traditional log analysis system, the proposed model has stronger semantic understanding ability, continuous learning ability and heterogeneous environment adaptability. Experiments on the cloud platform log dataset show that LLM-ID improves the fault location accuracy by 16.2%, which is significantly better than the current mainstream methods 

---
# Beyond Syntax: Action Semantics Learning for App Agents 

**Authors**: Bohan Tang, Dezhao Luo, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17697)  

**Abstract**: The advent of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with closed LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. With this insight, ASL employs a novel SEmantic Estimator (SEE) to compute a semantic reward to train the App agents in generating actions aligned with the semantics of ground truth actions, even when the syntactic forms differ. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments on offline and online smartphone App operation benchmarks show that ASL significantly improves the accuracy and generalisation of App agents over existing methods. 

---
# OmniReflect: Discovering Transferable Constitutions for LLM agents via Neuro-Symbolic Reflections 

**Authors**: Manasa Bharadwaj, Nikhil Verma, Kevin Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17449)  

**Abstract**: Efforts to improve Large Language Model (LLM) agent performance on complex tasks have largely focused on fine-tuning and iterative self-correction. However, these approaches often lack generalizable mechanisms for longterm learning and remain inefficient in dynamic environments. We introduce OmniReflect, a hierarchical, reflection-driven framework that constructs a constitution, a compact set of guiding principles distilled from task experiences, to enhance the effectiveness and efficiency of an LLM agent. OmniReflect operates in two modes: Self-sustaining, where a single agent periodically curates its own reflections during task execution, and Co-operative, where a Meta-advisor derives a constitution from a small calibration set to guide another agent. To construct these constitutional principles, we employ Neural, Symbolic, and NeuroSymbolic techniques, offering a balance between contextual adaptability and computational efficiency. Empirical results averaged across models show major improvements in task success, with absolute gains of +10.3% on ALFWorld, +23.8% on BabyAI, and +8.3% on PDDL in the Self-sustaining mode. Similar gains are seen in the Co-operative mode, where a lightweight Qwen3-4B ReAct agent outperforms all Reflexion baselines on BabyAI. These findings highlight the robustness and effectiveness of OmniReflect across environments and backbones. 

---
# Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories 

**Authors**: Islem Bouzenia, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18824)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly employed to automate complex software engineering tasks such as program repair and issue resolution. These agents operate by autonomously generating natural language thoughts, invoking external tools, and iteratively refining their solutions. Despite their widespread adoption, the internal decision-making processes of these agents remain largely unexplored, limiting our understanding of their operational dynamics and failure modes. In this paper, we present a large-scale empirical study of the thought-action-result trajectories of three state-of-the-art LLM-based agents: \textsc{RepairAgent}, \textsc{AutoCodeRover}, and \textsc{OpenHands}. We unify their interaction logs into a common format, capturing 120 trajectories and 2822 LLM interactions focused on program repair and issue resolution. Our study combines quantitative analyses of structural properties, action patterns, and token usage with qualitative assessments of reasoning coherence and feedback integration. We identify key trajectory characteristics such as iteration counts and token consumption, recurring action sequences, and the semantic coherence linking thoughts, actions, and their results. Our findings reveal behavioral motifs and anti-patterns that distinguish successful from failed executions, providing actionable insights for improving agent design, including prompting strategies, failure diagnosis, and anti-pattern detection. We release our dataset and annotation framework to support further research on transparent and robust autonomous software engineering agents. 

---
# Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown 

**Authors**: Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17589)  

**Abstract**: The real value of knowledge lies not just in its accumulation, but in its potential to be harnessed effectively to conquer the unknown. Although recent multimodal large language models (MLLMs) exhibit impressing multimodal capabilities, they often fail in rarely encountered domain-specific tasks due to limited relevant knowledge. To explore this, we adopt visual game cognition as a testbed and select Monster Hunter: World as the target to construct a multimodal knowledge graph (MH-MMKG), which incorporates multi-modalities and intricate entity relations. We also design a series of challenging queries based on MH-MMKG to evaluate the models' ability for complex knowledge retrieval and reasoning. Furthermore, we propose a multi-agent retriever that enables a model to autonomously search relevant knowledge without additional training. Experimental results show that our approach significantly enhances the performance of MLLMs, providing a new perspective on multimodal knowledge-augmented reasoning and laying a solid foundation for future research. 

---
# Security Assessment of DeepSeek and GPT Series Models against Jailbreak Attacks 

**Authors**: Xiaodong Wu, Xiangman Li, Jianbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.18543)  

**Abstract**: The widespread deployment of large language models (LLMs) has raised critical concerns over their vulnerability to jailbreak attacks, i.e., adversarial prompts that bypass alignment mechanisms and elicit harmful or policy-violating outputs. While proprietary models like GPT-4 have undergone extensive evaluation, the robustness of emerging open-source alternatives such as DeepSeek remains largely underexplored, despite their growing adoption in real-world applications. In this paper, we present the first systematic jailbreak evaluation of DeepSeek-series models, comparing them with GPT-3.5 and GPT-4 using the HarmBench benchmark. We evaluate seven representative attack strategies across 510 harmful behaviors categorized by both function and semantic domain. Our analysis reveals that DeepSeek's Mixture-of-Experts (MoE) architecture introduces routing sparsity that offers selective robustness against optimization-based attacks such as TAP-T, but leads to significantly higher vulnerability under prompt-based and manually engineered attacks. In contrast, GPT-4 Turbo demonstrates stronger and more consistent safety alignment across diverse behaviors, likely due to its dense Transformer design and reinforcement learning from human feedback. Fine-grained behavioral analysis and case studies further show that DeepSeek often routes adversarial prompts to under-aligned expert modules, resulting in inconsistent refusal behaviors. These findings highlight a fundamental trade-off between architectural efficiency and alignment generalization, emphasizing the need for targeted safety tuning and modular alignment strategies to ensure secure deployment of open-source LLMs. 

---
# Use Property-Based Testing to Bridge LLM Code Generation and Validation 

**Authors**: Lehan He, Zeren Chen, Zhe Zhang, Jing Shao, Xiang Gao, Lu Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18315)  

**Abstract**: Large Language Models (LLMs) excel at code generation, but ensuring their outputs to be functionally correct, especially in complex programming tasks, is a persistent challenge. While traditional Test-Driven Development (TDD) offers a path for code refinement, its efficacy with LLMs is often undermined by the scarcity of high-quality test cases or the pitfalls of automated test generation, including biased tests or inaccurate output predictions that can misdirect the correction process. This paper introduces Property-Generated Solver, a novel framework that leverages Property-Based Testing (PBT) to validate high-level program properties or invariants, instead of relying on specific input-output examples. These properties are often simpler to define and verify than directly predicting exhaustive test oracles, breaking the "cycle of self-deception" where tests might share flaws with the code they are meant to validate. Property-Generated Solver employs two collaborative LLM-based agents: a Generator dedicated to code generation and iterative refinement, and a Tester that manages the PBT life-cycle and formulate semantically rich feedback from property violations. The resulting comprehensive and actionable feedback then guides the Generator in its refinement efforts. By establishing PBT as the core validation engine within this iterative, closed-loop paradigm, Property-Generated Solver provides a robust mechanism for steering LLMs towards more correct and generalizable code. Extensive experimental results on multiple code generation benchmarks demonstrate that Property-Generated Solver achieves substantial pass@1 improvements, ranging from 23.1% to 37.3% relative gains over established TDD methods. 

---
# LOGICPO: Efficient Translation of NL-based Logical Problems to FOL using LLMs and Preference Optimization 

**Authors**: Koushik Viswanadha, Deepanway Ghosal, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2506.18383)  

**Abstract**: Logical reasoning is a key task for artificial intelligence due to it's role in major downstream tasks such as Question Answering, Summarization. Recent methods in improving the reasoning ability of LLMs fall short in correctly converting a natural language reasoning problem to an equivalent logical formulation, which hinders the framework's overall ability to reason. Towards this, we propose to use finetuning on a preference optimization dataset to learn to parse and represent a natural language problem as a whole to a consistent logical program by 1) introducing a new supervised and preference optimization dataset LogicPO, and 2) adopting popular techniques such as Direct Preference Optimization (DPO), Kahneman-Tversky optimization (KTO) to finetune open-source LLMs. Our best model with Phi-3.5 consistently outperforms GPT-3.5-turbo's (8-shot) by producing 10% more logically correct and with 14% less syntax errors. Through the framework and our improved evaluation metrics, we offer a promising direction in improving the logical reasoning of LLMs by better representing them in their logical formulations. 

---
# Understanding Reasoning in Thinking Language Models via Steering Vectors 

**Authors**: Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2506.18167)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of thinking language models that generate extensive internal reasoning chains before producing responses. While these models achieve improved performance, controlling their reasoning processes remains challenging. This work presents a steering approach for thinking LLMs by analyzing and manipulating specific reasoning behaviors in DeepSeek-R1-Distill models. Through a systematic experiment on 500 tasks across 10 diverse categories, we identify several reasoning behaviors exhibited by thinking models, including expressing uncertainty, generating examples for hypothesis validation, and backtracking in reasoning chains. We demonstrate that these behaviors are mediated by linear directions in the model's activation space and can be controlled using steering vectors. By extracting and applying these vectors, we provide a method to modulate specific aspects of the model's reasoning process, such as its tendency to backtrack or express uncertainty. Our approach offers practical tools for steering reasoning processes in thinking models in a controlled and interpretable manner. We validate our steering method using two DeepSeek-R1-Distill models, demonstrating consistent control across different model architectures. 

---
# Smart-LLaMA-DPO: Reinforced Large Language Model for Explainable Smart Contract Vulnerability Detection 

**Authors**: Lei Yu, Zhirong Huang, Hang Yuan, Shiqi Cheng, Li Yang, Fengjun Zhang, Chenjie Shen, Jiajia Ma, Jingyuan Zhang, Junyi Lu, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18245)  

**Abstract**: Smart contract vulnerability detection remains a major challenge in blockchain security. Existing vulnerability detection methods face two main issues: (1) Existing datasets lack comprehensive coverage and high-quality explanations for preference learning. (2) Large language models (LLMs) often struggle with accurately interpreting specific concepts in smart contract security. Empirical analysis shows that even after continual pre-training (CPT) and supervised fine-tuning (SFT), LLMs may misinterpret the execution order of state changes, resulting in incorrect explanations despite making correct detection decisions. To address these challenges, we propose Smart-LLaMA-DPO based on LLaMA-3.1-8B. We construct a comprehensive dataset covering four major vulnerability types and machine-unauditable vulnerabilities, including precise labels, explanations, and locations for SFT, as well as high-quality and low-quality output pairs for Direct Preference Optimization (DPO). Second, we perform CPT using large-scale smart contract to enhance the LLM's understanding of specific security practices in smart contracts. Futhermore, we conduct SFT with our comprehensive dataset. Finally, we apply DPO, leveraging human feedback and a specially designed loss function that increases the probability of preferred explanations while reducing the likelihood of non-preferred outputs. We evaluate Smart-LLaMA-DPO on four major vulnerability types: reentrancy, timestamp dependence, integer overflow/underflow, and delegatecall, as well as machine-unauditable vulnerabilities. Our method significantly outperforms state-of-the-art baselines, with average improvements of 10.43% in F1 score and 7.87% in accuracy. Moreover, both LLM evaluation and human evaluation confirm that our method generates more correct, thorough, and clear explanations. 

---
# Mechanistic Interpretability in the Presence of Architectural Obfuscation 

**Authors**: Marcos Florencio, Thomas Barton  

**Link**: [PDF](https://arxiv.org/pdf/2506.18053)  

**Abstract**: Architectural obfuscation - e.g., permuting hidden-state tensors, linearly transforming embedding tables, or remapping tokens - has recently gained traction as a lightweight substitute for heavyweight cryptography in privacy-preserving large-language-model (LLM) inference. While recent work has shown that these techniques can be broken under dedicated reconstruction attacks, their impact on mechanistic interpretability has not been systematically studied. In particular, it remains unclear whether scrambling a network's internal representations truly thwarts efforts to understand how the model works, or simply relocates the same circuits to an unfamiliar coordinate system. We address this gap by analyzing a GPT-2-small model trained from scratch with a representative obfuscation map. Assuming the obfuscation map is private and the original basis is hidden (mirroring an honest-but-curious server), we apply logit-lens attribution, causal path-patching, and attention-head ablation to locate and manipulate known circuits. Our findings reveal that obfuscation dramatically alters activation patterns within attention heads yet preserves the layer-wise computational graph. This disconnect hampers reverse-engineering of user prompts: causal traces lose their alignment with baseline semantics, and token-level logit attributions become too noisy to reconstruct. At the same time, feed-forward and residual pathways remain functionally intact, suggesting that obfuscation degrades fine-grained interpretability without compromising top-level task performance. These results establish quantitative evidence that architectural obfuscation can simultaneously (i) retain global model behaviour and (ii) impede mechanistic analyses of user-specific content. By mapping where interpretability breaks down, our study provides guidance for future privacy defences and for robustness-aware interpretability tooling. 

---
# Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster 

**Authors**: Fenghe Tang, Wenxin Ma, Zhiyang He, Xiaodong Tao, Zihang Jiang, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18034)  

**Abstract**: With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid structure that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder segmentation framework (LLM4Seg). Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek. 

---
# Federated Learning-Based Data Collaboration Method for Enhancing Edge Cloud AI System Security Using Large Language Models 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.18087)  

**Abstract**: With the widespread application of edge computing and cloud systems in AI-driven applications, how to maintain efficient performance while ensuring data privacy has become an urgent security issue. This paper proposes a federated learning-based data collaboration method to improve the security of edge cloud AI systems, and use large-scale language models (LLMs) to enhance data privacy protection and system robustness. Based on the existing federated learning framework, this method introduces a secure multi-party computation protocol, which optimizes the data aggregation and encryption process between distributed nodes by using LLM to ensure data privacy and improve system efficiency. By combining advanced adversarial training techniques, the model enhances the resistance of edge cloud AI systems to security threats such as data leakage and model poisoning. Experimental results show that the proposed method is 15% better than the traditional federated learning method in terms of data protection and model robustness. 

---
# SurgVidLM: Towards Multi-grained Surgical Video Understanding with Large Language Model 

**Authors**: Guankun Wang, Wenjin Mo, Junyi Wang, Long Bai, Kun Yuan, Ming Hu, Jinlin Wu, Junjun He, Yiming Huang, Nicolas Padoy, Zhen Lei, Hongbin Liu, Nassir Navab, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.17873)  

**Abstract**: Recent advances in Multimodal Large Language Models have demonstrated great potential in the medical domain, facilitating users to understand surgical scenes and procedures. Beyond image-based methods, the exploration of Video Large Language Models (Vid-LLMs) has emerged as a promising avenue for capturing the complex sequences of information involved in surgery. However, there is still a lack of Vid-LLMs specialized for fine-grained surgical video understanding tasks, which is crucial for analyzing specific processes or details within a surgical procedure. To bridge this gap, we propose SurgVidLM, the first video language model designed to address both full and fine-grained surgical video comprehension. To train our SurgVidLM, we construct the SVU-31K dataset which consists of over 31K video-instruction pairs, enabling both holistic understanding and detailed analysis of surgical procedures. Furthermore, we introduce the StageFocus mechanism which is a two-stage framework performing the multi-grained, progressive understanding of surgical videos. We also develop the Multi-frequency Fusion Attention to effectively integrate low and high-frequency visual tokens, ensuring the retention of critical information. Experimental results demonstrate that SurgVidLM significantly outperforms state-of-the-art Vid-LLMs in both full and fine-grained video understanding tasks, showcasing its superior capability in capturing complex procedural contexts. 

---
# Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models 

**Authors**: Jihyun Kim, Junho Park, Kyeongbo Kong, Suk-Ju Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17707)  

**Abstract**: We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. For precise control of a room's each attribute, we decompose the challenging task into simpler steps such as creating plausible 3D coordinates for room meshes, generating panorama images for the texture, constructing 3D meshes by integrating the coordinates and panorama texture images, and arranging furniture. To support the various decomposed tasks with a unified framework, we incorporate visual programming (VP). VP is a method that utilizes a large language model (LLM) to write a Python-like program which is an ordered list of necessary modules for the various tasks given in natural language. We develop most of the modules. Especially, for the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we enhance the panorama image generation quality by optimizing the training objective with a 1D representation of a panorama scene obtained from bidirectional LSTM. We demonstrate Programmable-Room's flexibility in generating and editing 3D room meshes, and prove our framework's superiority to an existing model quantitatively and qualitatively. Project page is available in this https URL. 

---
# LLM-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting 

**Authors**: Zesen Wang, Yonggang Li, Lijuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17631)  

**Abstract**: Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting. 

---
# Distilling On-device Language Models for Robot Planning with Minimal Human Intervention 

**Authors**: Zachary Ravichandran, Ignacio Hounie, Fernando Cladera, Alejandro Ribeiro, George J. Pappas, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17486)  

**Abstract**: Large language models (LLMs) provide robots with powerful contextual reasoning abilities and a natural human interface. Yet, current LLM-enabled robots typically depend on cloud-hosted models, limiting their usability in environments with unreliable communication infrastructure, such as outdoor or industrial settings. We present PRISM, a framework for distilling small language model (SLM)-enabled robot planners that run on-device with minimal human supervision. Starting from an existing LLM-enabled planner, PRISM automatically synthesizes diverse tasks and environments, elicits plans from the LLM, and uses this synthetic dataset to distill a compact SLM as a drop-in replacement of the source model. We apply PRISM to three LLM-enabled planners for mapping and exploration, manipulation, and household assistance, and we demonstrate that PRISM improves the performance of Llama-3.2-3B from 10-20% of GPT-4o's performance to over 93% - using only synthetic data. We further demonstrate that the distilled planners generalize across heterogeneous robotic platforms (ground and aerial) and diverse environments (indoor and outdoor). We release all software, trained models, and datasets at this https URL. 

---
# Re-Evaluating Code LLM Benchmarks Under Semantic Mutation 

**Authors**: Zhiyuan Pan, Xing Hu, Xin Xia, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17369)  

**Abstract**: In the era of large language models (LLMs), code benchmarks have become an important research area in software engineering and are widely used by practitioners. These benchmarks evaluate the performance of LLMs on specific code-related tasks, such as code understanding and generation. A critical step in constructing code benchmarks is the design of prompts. However, as existing code benchmarks typically rely on a single prompt template per task, they are prone to the issue of prompt sensitivity, where minor prompt variations could result in substantial performance variations, leading to unreliable evaluations of model capabilities.
While previous studies have explored prompt sensitivity, their experimental designs and findings are limited to traditional natural language processing (NLP) tasks. In this paper, we present an empirical study to investigate prompt sensitivity in code benchmarks. We first propose a general framework that modifies prompt templates in a manner that preserves both their semantics and their structure as much as possible. Based on the framework, we conduct extensive experiments across eight code benchmark tasks on 10 representative open-source LLMs, with each task featuring 100 semantically similar prompt templates. We then analyze the evaluation results using various statistical metrics, focusing on both absolute and relative model performance. Our findings suggest that even slight prompt variations can lead to significant shifts in performance. Additionally, we observe that such variations can introduce inconsistencies in the performance rankings across different models. These insights highlight the need for considering prompt sensitivity when designing future code benchmarks, to ensure more reliable and accurate evaluation of LLM capabilities. 

---
# SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification 

**Authors**: Zhenglin Lai, Mengyao Liao, Dong Xu, Zebin Zhao, Zhihang Yuan, Chao Fan, Jianqiang Li, Bingzhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17368)  

**Abstract**: Large language models based on Mixture-of-Experts have achieved substantial gains in efficiency and scalability, yet their architectural uniqueness introduces underexplored safety alignment challenges. Existing safety alignment strategies, predominantly designed for dense models, are ill-suited to address MoE-specific vulnerabilities. In this work, we formalize and systematically study MoE model's positional vulnerability - the phenomenon where safety-aligned behaviors rely on specific expert modules, revealing critical risks inherent to MoE architectures. To this end, we present SAFEx, an analytical framework that robustly identifies, characterizes, and validates the safety-critical experts using a novel Stability-based Expert Selection (SES) algorithm. Notably, our approach enables the explicit decomposition of safety-critical experts into distinct functional groups, including those responsible for harmful content detection and those controlling safe response generation. Extensive experiments on mainstream MoE models, such as the recently released Qwen3-MoE, demonstrated that their intrinsic safety mechanisms heavily rely on a small subset of positional experts. Disabling these experts significantly compromised the models' ability to refuse harmful requests. For Qwen3-MoE with 6144 experts (in the FNN layer), we find that disabling as few as 12 identified safety-critical experts can cause the refusal rate to drop by 22%, demonstrating the disproportionate impact of a small set of experts on overall model safety. 

---
# A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant 

**Authors**: Sunjun Kweon, Sooyohn Nam, Hyunseung Lim, Hwajung Hong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17363)  

**Abstract**: Virtual Teaching Assistants (VTAs) powered by Large Language Models (LLMs) have the potential to enhance student learning by providing instant feedback and facilitating multi-turn interactions. However, empirical studies on their effectiveness and acceptance in real-world classrooms are limited, leaving their practical impact uncertain. In this study, we develop an LLM-based VTA and deploy it in an introductory AI programming course with 477 graduate students. To assess how student perceptions of the VTA's performance evolve over time, we conduct three rounds of comprehensive surveys at different stages of the course. Additionally, we analyze 3,869 student--VTA interaction pairs to identify common question types and engagement patterns. We then compare these interactions with traditional student--human instructor interactions to evaluate the VTA's role in the learning process. Through a large-scale empirical study and interaction analysis, we assess the feasibility of deploying VTAs in real-world classrooms and identify key challenges for broader adoption. Finally, we release the source code of our VTA system, fostering future advancements in AI-driven education: \texttt{this https URL}. 

---
# Automatic Large Language Models Creation of Interactive Learning Lessons 

**Authors**: Jionghao Lin, Jiarui Rao, Yiyang Zhao, Yuting Wang, Ashish Gurung, Amanda Barany, Jaclyn Ocumpaugh, Ryan S. Baker, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17356)  

**Abstract**: We explore the automatic generation of interactive, scenario-based lessons designed to train novice human tutors who teach middle school mathematics online. Employing prompt engineering through a Retrieval-Augmented Generation approach with GPT-4o, we developed a system capable of creating structured tutor training lessons. Our study generated lessons in English for three key topics: Encouraging Students' Independence, Encouraging Help-Seeking Behavior, and Turning on Cameras, using a task decomposition prompting strategy that breaks lesson generation into sub-tasks. The generated lessons were evaluated by two human evaluators, who provided both quantitative and qualitative evaluations using a comprehensive rubric informed by lesson design research. Results demonstrate that the task decomposition strategy led to higher-rated lessons compared to single-step generation. Human evaluators identified several strengths in the LLM-generated lessons, including well-structured content and time-saving potential, while also noting limitations such as generic feedback and a lack of clarity in some instructional sections. These findings underscore the potential of hybrid human-AI approaches for generating effective lessons in tutor training. 

---
# Differentiation-Based Extraction of Proprietary Data from Fine-Tuned LLMs 

**Authors**: Zongjie Li, Daoyuan Wu, Shuai Wang, Zhendong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17353)  

**Abstract**: The increasing demand for domain-specific and human-aligned Large Language Models (LLMs) has led to the widespread adoption of Supervised Fine-Tuning (SFT) techniques. SFT datasets often comprise valuable instruction-response pairs, making them highly valuable targets for potential extraction. This paper studies this critical research problem for the first time. We start by formally defining and formulating the problem, then explore various attack goals, types, and variants based on the unique properties of SFT data in real-world scenarios. Based on our analysis of extraction behaviors of direct extraction, we develop a novel extraction method specifically designed for SFT models, called Differentiated Data Extraction (DDE), which exploits the confidence levels of fine-tuned models and their behavioral differences from pre-trained base models. Through extensive experiments across multiple domains and scenarios, we demonstrate the feasibility of SFT data extraction using DDE. Our results show that DDE consistently outperforms existing extraction baselines in all attack settings. To counter this new attack, we propose a defense mechanism that mitigates DDE attacks with minimal impact on model performance. Overall, our research reveals hidden data leak risks in fine-tuned LLMs and provides insights for developing more secure models. 

---
# I Know Which LLM Wrote Your Code Last Summer: LLM generated Code Stylometry for Authorship Attribution 

**Authors**: Tamas Bisztray, Bilel Cherif, Richard A. Dubniczky, Nils Gruschka, Bertalan Borsos, Mohamed Amine Ferrag, Attila Kovacs, Vasileios Mavroeidis, Norbert Tihanyi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17323)  

**Abstract**: Detecting AI-generated code, deepfakes, and other synthetic content is an emerging research challenge. As code generated by Large Language Models (LLMs) becomes more common, identifying the specific model behind each sample is increasingly important. This paper presents the first systematic study of LLM authorship attribution for C programs. We released CodeT5-Authorship, a novel model that uses only the encoder layers from the original CodeT5 encoder-decoder architecture, discarding the decoder to focus on classification. Our model's encoder output (first token) is passed through a two-layer classification head with GELU activation and dropout, producing a probability distribution over possible authors. To evaluate our approach, we introduce LLM-AuthorBench, a benchmark of 32,000 compilable C programs generated by eight state-of-the-art LLMs across diverse tasks. We compare our model to seven traditional ML classifiers and eight fine-tuned transformer models, including BERT, RoBERTa, CodeBERT, ModernBERT, DistilBERT, DeBERTa-V3, Longformer, and LoRA-fine-tuned Qwen2-1.5B. In binary classification, our model achieves 97.56% accuracy in distinguishing C programs generated by closely related models such as GPT-4.1 and GPT-4o, and 95.40% accuracy for multi-class attribution among five leading LLMs (Gemini 2.5 Flash, Claude 3.5 Haiku, GPT-4.1, Llama 3.3, and DeepSeek-V3). To support open science, we release the CodeT5-Authorship architecture, the LLM-AuthorBench benchmark, and all relevant Google Colab scripts on GitHub: this https URL. 

---
# LLM Jailbreak Oracle 

**Authors**: Shuyi Lin, Anshuman Suri, Alina Oprea, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17299)  

**Abstract**: As large language models (LLMs) become increasingly deployed in safety-critical applications, the lack of systematic methods to assess their vulnerability to jailbreak attacks presents a critical security gap. We introduce the jailbreak oracle problem: given a model, prompt, and decoding strategy, determine whether a jailbreak response can be generated with likelihood exceeding a specified threshold. This formalization enables a principled study of jailbreak vulnerabilities. Answering the jailbreak oracle problem poses significant computational challenges -- the search space grows exponentially with the length of the response tokens. We present Boa, the first efficient algorithm for solving the jailbreak oracle problem. Boa employs a three-phase search strategy: (1) constructing block lists to identify refusal patterns, (2) breadth-first sampling to identify easily accessible jailbreaks, and (3) depth-first priority search guided by fine-grained safety scores to systematically explore promising low-probability paths. Boa enables rigorous security assessments including systematic defense evaluation, standardized comparison of red team attacks, and model certification under extreme adversarial conditions. 

---
# Step-by-Step Reasoning Attack: Revealing 'Erased' Knowledge in Large Language Models 

**Authors**: Yash Sinha, Manit Baser, Murari Mandal, Dinil Mon Divakaran, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.17279)  

**Abstract**: Knowledge erasure in large language models (LLMs) is important for ensuring compliance with data and AI regulations, safeguarding user privacy, mitigating bias, and misinformation. Existing unlearning methods aim to make the process of knowledge erasure more efficient and effective by removing specific knowledge while preserving overall model performance, especially for retained information. However, it has been observed that the unlearning techniques tend to suppress and leave the knowledge beneath the surface, thus making it retrievable with the right prompts. In this work, we demonstrate that \textit{step-by-step reasoning} can serve as a backdoor to recover this hidden information. We introduce a step-by-step reasoning-based black-box attack, Sleek, that systematically exposes unlearning failures. We employ a structured attack framework with three core components: (1) an adversarial prompt generation strategy leveraging step-by-step reasoning built from LLM-generated queries, (2) an attack mechanism that successfully recalls erased content, and exposes unfair suppression of knowledge intended for retention and (3) a categorization of prompts as direct, indirect, and implied, to identify which query types most effectively exploit unlearning weaknesses. Through extensive evaluations on four state-of-the-art unlearning techniques and two widely used LLMs, we show that existing approaches fail to ensure reliable knowledge removal. Of the generated adversarial prompts, 62.5% successfully retrieved forgotten Harry Potter facts from WHP-unlearned Llama, while 50% exposed unfair suppression of retained knowledge. Our work highlights the persistent risks of information leakage, emphasizing the need for more robust unlearning strategies for erasure. 

---
# LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research 

**Authors**: Shuo Yan, Ruochen Li, Ziming Luo, Zimu Wang, Daoyang Li, Liqiang Jing, Kaiyu He, Peilin Wu, George Michalopoulos, Yue Zhang, Ziyang Zhang, Mian Zhang, Zhiyu Chen, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.17335)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present LMR-BENCH, a benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from Language Modeling Research. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked functions, and instructions for implementing these functions. We conduct extensive experiments in standard prompting and LLM agent settings with state-of-the-art LLMs, evaluating the accuracy of unit tests and performing LLM-based evaluation of code correctness. Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLM agents' ability to autonomously reproduce scientific research 

---
# Does Multimodal Large Language Model Truly Unlearn? Stealthy MLLM Unlearning Attack 

**Authors**: Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17265)  

**Abstract**: Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the ``forget'' sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior. 

---
# OAT-Rephrase: Optimization-Aware Training Data Rephrasing for Zeroth-Order LLM Fine-Tuning 

**Authors**: Jikai Long, Zijian Hu, Xiaodong Yu, Jianwen Xie, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17264)  

**Abstract**: Fine-tuning large language models (LLMs) using zeroth-order optimization (ZO) offers a memory-efficient alternative to gradient-based methods but suffers from slower convergence and unstable optimization due to noisy gradient estimates. This paper introduces OAT-Rephrase, an Optimization-Aware Training data rephrasing strategy that leverages an LLM to rephrase training instances based on its understanding of the ZO dynamics, specifically MeZO, derived directly from its paper. The approach incorporates a dual-stage pipeline featuring a rewriter LLM and a semantic judge, ensuring all rephrasings retain task relevance and logical consistency. Evaluations across five classification tasks and three LLM architectures demonstrate that OAT-Rephrase consistently improves MeZO fine-tuning performance, often narrowing or eliminating the gap with first-order methods. Our findings suggest that optimization-aware rephrasing serves as a reusable and low-overhead enhancement for zeroth-order tuning regimes. 

---
# UltraSketchLLM: Saliency-Driven Sketching for Ultra-Low Bit LLM Compression 

**Authors**: Sunan Zou, Ziyun Zhang, Xueting Sun, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17255)  

**Abstract**: The rapid growth of large language models (LLMs) has outpaced the memory constraints of edge devices, necessitating extreme weight compression beyond the 1-bit limit. While quantization reduces model size, it is fundamentally limited to 1 bit per weight. Existing multiple-to-one compression methods either rely on mapping tables (inducing memory overhead) or incur severe accuracy degradation due to random weight grouping. We introduce UltraSketchLLM, an index-free, sketch-based framework that achieves ultra-low bit compression (down to 0.5 bits per weight) while preserving model performance. UltraSketchLLM leverages data sketching, a sub-linear representation technique from streaming applications, to map multiple weights to single values with bounded error. Our approach integrates an underestimate AbsMaxMin sketch to minimize relative errors for small weights, importance-aware space allocation to prioritize salient weights, and a straight-through estimator for compression-aware finetuning. Experiments on Llama-3.2-1B demonstrate up to 0.5-bit compression with competitive perplexity, alongside tolerable latency overhead. UltraSketchLLM offers a practical solution for deploying LLMs in resource-constrained environments. 

---
# Training-free LLM Verification via Recycling Few-shot Examples 

**Authors**: Dongseok Lee, Jimyung Hong, Dongyoung Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.17251)  

**Abstract**: Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges. Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (Referi). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, Referi evaluates the generated outputs by combining two different scores, designed motivated from Bayes' rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.8%-through effective response selection, without additional training. 

---
# Adaptive Sample Scheduling for Direct Preference Optimization 

**Authors**: Zixuan Huang, Yikun Ban, Lean Fu, Xiaojie Li, Zhongxiang Dai, Jianxin Li, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17252)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for aligning large language models (LLMs) with human preferences. However, its performance is highly dependent on the quality of the underlying human preference data. To address this bottleneck, prior work has explored various data selection strategies, but these methods often overlook the impact of the evolving states of the language model during the DPO process. %including active querying, response pair selection, and data pre-selection. In this paper, we introduce a novel problem: Sample Scheduling for DPO, which aims to dynamically and adaptively schedule training samples based on the model's evolving states throughout preference optimization. To solve this problem, we propose SamS, an efficient and effective algorithm that adaptively selects samples in each training batch based on the LLM's learning feedback to maximize the potential generalization performance. Notably, without modifying the core DPO algorithm, simply integrating SamS significantly improves performance across tasks, with minimal additional computational overhead. This work points to a promising new direction for improving LLM alignment through more effective utilization of fixed preference datasets. 

---
# Keeping Up with the Models: Online Deployment and Routing of LLMs at Scale 

**Authors**: Shaoang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17254)  

**Abstract**: The rapid pace at which new large language models (LLMs) appear -- and older ones become obsolete -- forces LLM service providers to juggle a streaming inventory of models while respecting tight deployment capacity and per-query cost budgets. We cast the reality as an online decision problem that couples stage-wise deployment, made at fixed maintenance windows, with per-query routing among the models kept live. We introduce StageRoute, a hierarchical algorithm that (i) optimistically selects up to $M_max$ models for the next stage using reward upper-confidence and cost lower-confidence bounds, then (ii) solves a budget-constrained bandit sub-problem to route each incoming query. We prove that StageRoute achieves a regret of order $T^{2/3}$ and provide a matching lower bound, thereby establishing its near-optimality. Moreover, our experiments confirm the theory, demonstrating that StageRoute performs close to the optimum in practical settings. 

---
