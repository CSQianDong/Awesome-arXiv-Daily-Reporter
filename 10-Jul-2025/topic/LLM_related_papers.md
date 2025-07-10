# CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.06715)  

**Abstract**: Large language models (LLMs), including zero-shot and few-shot paradigms, have shown promising capabilities in clinical text generation. However, real-world applications face two key challenges: (1) patient data is highly unstructured, heterogeneous, and scattered across multiple note types and (2) clinical notes are often long and semantically dense, making naive prompting infeasible due to context length constraints and the risk of omitting clinically relevant information.
We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a domain-specific framework for structured and clinically grounded text generation using LLMs. It incorporates a novel hierarchical chunking strategy that respects clinical document structure and introduces a task-specific dual-stage retrieval mechanism. The global stage identifies relevant note types using evidence-based queries, while the local stage extracts high-value content within those notes creating relevance at both document and section levels.
We apply the system to generate structured progress notes for individual hospital visits using 15 clinical note types from the MIMIC-III dataset. Experiments show that it preserves temporal and semantic alignment across visits, achieving an average alignment score of 87.7%, surpassing the 80.7% baseline from real clinician-authored notes. The generated outputs also demonstrate high consistency across LLMs, reinforcing deterministic behavior essential for reproducibility, reliability, and clinical trust. 

---
# Boosting Parameter Efficiency in LLM-Based Recommendation through Sophisticated Pruning 

**Authors**: Shanle Zheng, Keqin Bao, Jizhi Zhang, Yang Zhang, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2507.07064)  

**Abstract**: LLM-based recommender systems have made significant progress; however, the deployment cost associated with the large parameter volume of LLMs still hinders their real-world applications. This work explores parameter pruning to improve parameter efficiency while maintaining recommendation quality, thereby enabling easier deployment. Unlike existing approaches that focus primarily on inter-layer redundancy, we uncover intra-layer redundancy within components such as self-attention and MLP modules. Building on this analysis, we propose a more fine-grained pruning approach that integrates both intra-layer and layer-wise pruning. Specifically, we introduce a three-stage pruning strategy that progressively prunes parameters at different levels and parts of the model, moving from intra-layer to layer-wise pruning, or from width to depth. Each stage also includes a performance restoration step using distillation techniques, helping to strike a balance between performance and parameter efficiency. Empirical results demonstrate the effectiveness of our approach: across three datasets, our models achieve an average of 88% of the original model's performance while pruning more than 95% of the non-embedding parameters. This underscores the potential of our method to significantly reduce resource requirements without greatly compromising recommendation quality. Our code will be available at: this https URL 

---
# GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models 

**Authors**: Zhen Yang, Haitao Lin, Jiawei xue, Ziji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06507)  

**Abstract**: In the past year, Generative Recommendations (GRs) have undergone substantial advancements, especially in leveraging the powerful sequence modeling and reasoning capabilities of Large Language Models (LLMs) to enhance overall recommendation performance. LLM-based GRs are forming a new paradigm that is distinctly different from discriminative recommendations, showing strong potential to replace traditional recommendation systems heavily dependent on complex hand-crafted features. In this paper, we provide a comprehensive survey aimed at facilitating further research of LLM-based GRs. Initially, we outline the general preliminaries and application cases of LLM-based GRs. Subsequently, we introduce the main considerations when LLM-based GRs are applied in real industrial scenarios. Finally, we explore promising directions for LLM-based GRs. We hope that this survey contributes to the ongoing advancement of the GR domain. 

---
# MultiJustice: A Chinese Dataset for Multi-Party, Multi-Charge Legal Prediction 

**Authors**: Xiao Wang, Jiahuan Pei, Diancheng Shui, Zhiguang Han, Xin Sun, Dawei Zhu, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06909)  

**Abstract**: Legal judgment prediction offers a compelling method to aid legal practitioners and researchers. However, the research question remains relatively under-explored: Should multiple defendants and charges be treated separately in LJP? To address this, we introduce a new dataset namely multi-person multi-charge prediction (MPMCP), and seek the answer by evaluating the performance of several prevailing legal large language models (LLMs) on four practical legal judgment scenarios: (S1) single defendant with a single charge, (S2) single defendant with multiple charges, (S3) multiple defendants with a single charge, and (S4) multiple defendants with multiple charges. We evaluate the dataset across two LJP tasks, i.e., charge prediction and penalty term prediction. We have conducted extensive experiments and found that the scenario involving multiple defendants and multiple charges (S4) poses the greatest challenges, followed by S2, S3, and S1. The impact varies significantly depending on the model. For example, in S4 compared to S1, InternLM2 achieves approximately 4.5% lower F1-score and 2.8% higher LogD, while Lawformer demonstrates around 19.7% lower F1-score and 19.0% higher LogD. Our dataset and code are available at this https URL. 

---
# Adaptive Termination for Multi-round Parallel Reasoning: An Universal Semantic Entropy-Guided Framework 

**Authors**: Zenan Xu, Zexuan Qiu, Guanhua Huang, Kun Li, Siheng Li, Chenchen Zhang, Kejiao Li, Qi Yi, Yuhao Jiang, Bo Zhou, Fengzong Lian, Zhanhui Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06829)  

**Abstract**: Recent advances in large language models (LLMs) have accelerated progress toward artificial general intelligence, with inference-time scaling emerging as a key technique. Contemporary approaches leverage either sequential reasoning (iteratively extending chains of thought) or parallel reasoning (generating multiple solutions simultaneously) to scale inference. However, both paradigms face fundamental limitations: sequential scaling typically relies on arbitrary token budgets for termination, leading to inefficiency or premature cutoff; while parallel scaling often lacks coordination among parallel branches and requires intrusive fine-tuning to perform effectively. In light of these challenges, we aim to design a flexible test-time collaborative inference framework that exploits the complementary strengths of both sequential and parallel reasoning paradigms. Towards this goal, the core challenge lies in developing an efficient and accurate intrinsic quality metric to assess model responses during collaborative inference, enabling dynamic control and early termination of the reasoning trace. To address this challenge, we introduce semantic entropy (SE), which quantifies the semantic diversity of parallel model responses and serves as a robust indicator of reasoning quality due to its strong negative correlation with accuracy... 

---
# Checklist Engineering Empowers Multilingual LLM Judges 

**Authors**: Mohammad Ghiasvand Mohammadkhani, Hamid Beigy  

**Link**: [PDF](https://arxiv.org/pdf/2507.06774)  

**Abstract**: Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model. 

---
# Shifting from Ranking to Set Selection for Retrieval Augmented Generation 

**Authors**: Dahyun Lee, Yongrae Jo, Haeju Park, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.06838)  

**Abstract**: Retrieval in Retrieval-Augmented Generation(RAG) must ensure that retrieved passages are not only individually relevant but also collectively form a comprehensive set. Existing approaches primarily rerank top-k passages based on their individual relevance, often failing to meet the information needs of complex queries in multi-hop question answering. In this work, we propose a set-wise passage selection approach and introduce SETR, which explicitly identifies the information requirements of a query through Chain-of-Thought reasoning and selects an optimal set of passages that collectively satisfy those requirements. Experiments on multi-hop RAG benchmarks show that SETR outperforms both proprietary LLM-based rerankers and open-source baselines in terms of answer correctness and retrieval quality, providing an effective and efficient alternative to traditional rerankers in RAG systems. The code is available at this https URL 

---
# The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production 

**Authors**: Juan B. Gutiérrez  

**Link**: [PDF](https://arxiv.org/pdf/2507.06565)  

**Abstract**: Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest. 

---
# Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation 

**Authors**: Liliang Ren, Congcong Chen, Haoran Xu, Young Jin Kim, Adam Atkinson, Zheng Zhan, Jiankai Sun, Baolin Peng, Liyuan Liu, Shuohang Wang, Hao Cheng, Jianfeng Gao, Weizhu Chen, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06607)  

**Abstract**: Recent advances in language modeling have demonstrated the effectiveness of State Space Models (SSMs) for efficient sequence modeling. While hybrid architectures such as Samba and the decoder-decoder architecture, YOCO, have shown promising performance gains over Transformers, prior works have not investigated the efficiency potential of representation sharing between SSM layers. In this paper, we introduce the Gated Memory Unit (GMU), a simple yet effective mechanism for efficient memory sharing across layers. We apply it to create SambaY, a decoder-hybrid-decoder architecture that incorporates GMUs in the cross-decoder to share memory readout states from a Samba-based self-decoder. SambaY significantly enhances decoding efficiency, preserves linear pre-filling time complexity, and boosts long-context performance, all while eliminating the need for explicit positional encoding. Through extensive scaling experiments, we demonstrate that our model exhibits a significantly lower irreducible loss compared to a strong YOCO baseline, indicating superior performance scalability under large-scale compute regimes. Our largest model enhanced with Differential Attention, Phi4-mini-Flash-Reasoning, achieves significantly better performance than Phi4-mini-Reasoning on reasoning tasks such as Math500, AIME24/25, and GPQA Diamond without any reinforcement learning, while delivering up to 10x higher decoding throughput on 2K-length prompts with 32K generation length under the vLLM inference framework. We release our training codebase on open-source data at this https URL. 

---
# InvestAlign: Overcoming Data Scarcity in Aligning Large Language Models with Investor Decision-Making Processes under Herd Behavior 

**Authors**: Huisheng Wang, Zhuoshi Pan, Hangjing Zhang, Mingxiao Liu, Hanqing Gao, H. Vicky Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06528)  

**Abstract**: Aligning Large Language Models (LLMs) with investor decision-making processes under herd behavior is a critical challenge in behavioral finance, which grapples with a fundamental limitation: the scarcity of real-user data needed for Supervised Fine-Tuning (SFT). While SFT can bridge the gap between LLM outputs and human behavioral patterns, its reliance on massive authentic data imposes substantial collection costs and privacy risks. We propose InvestAlign, a novel framework that constructs high-quality SFT datasets by leveraging theoretical solutions to similar and simple optimal investment problems rather than complex scenarios. Our theoretical analysis demonstrates that training LLMs with InvestAlign-generated data achieves faster parameter convergence than using real-user data, suggesting superior learning efficiency. Furthermore, we develop InvestAgent, an LLM agent fine-tuned with InvestAlign, which demonstrates significantly closer alignment to real-user data than pre-SFT models in both simple and complex investment problems. This highlights our proposed InvestAlign as a promising approach with the potential to address complex optimal investment problems and align LLMs with investor decision-making processes under herd behavior. Our code is publicly available at this https URL. 

---
# Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining: Method, Evaluation and Applications 

**Authors**: Seonwu Kim, Yohan Na, Kihun Kim, Hanhee Cho, Geun Lim, Mintae Kim, Seongik Park, Ki Hyun Kim, Youngsub Han, Byoung-Ki Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2507.06795)  

**Abstract**: The emergence of open-source large language models (LLMs) has expanded opportunities for enterprise applications; however, many organizations still lack the infrastructure to deploy and maintain large-scale models. As a result, small LLMs (sLLMs) have become a practical alternative, despite their inherent performance limitations. While Domain Adaptive Continual Pretraining (DACP) has been previously explored as a method for domain adaptation, its utility in commercial applications remains under-examined. In this study, we validate the effectiveness of applying a DACP-based recipe across diverse foundation models and service domains. Through extensive experiments and real-world evaluations, we demonstrate that DACP-applied sLLMs achieve substantial gains in target domain performance while preserving general capabilities, offering a cost-efficient and scalable solution for enterprise-level deployment. 

---
# Pun Intended: Multi-Agent Translation of Wordplay with Contrastive Learning and Phonetic-Semantic Embeddings 

**Authors**: Russell Taylor, Benjamin Herbert, Michael Sana  

**Link**: [PDF](https://arxiv.org/pdf/2507.06506)  

**Abstract**: Translating wordplay across languages presents unique challenges that have long confounded both professional human translators and machine translation systems. This research proposes a novel approach for translating puns from English to French by combining state-of-the-art large language models with specialized techniques for wordplay generation.
Our methodology employs a three-stage approach. First, we establish a baseline using multiple frontier large language models with feedback based on a new contrastive learning dataset. Second, we implement a guided chain-of-thought pipeline with combined phonetic-semantic embeddings. Third, we implement a multi-agent generator-discriminator framework for evaluating and regenerating puns with feedback.
Moving beyond the limitations of literal translation, our methodology's primary objective is to capture the linguistic creativity and humor of the source text wordplay, rather than simply duplicating its vocabulary. Our best runs earned first and second place in the CLEF JOKER 2025 Task 2 competition where they were evaluated manually by expert native French speakers.
This research addresses a gap between translation studies and computational linguistics by implementing linguistically-informed techniques for wordplay translation, advancing our understanding of how language models can be leveraged to handle the complex interplay between semantic ambiguity, phonetic similarity, and the implicit cultural and linguistic awareness needed for successful humor. 

---
# Text to model via SysML: Automated generation of dynamical system computational models from unstructured natural language text via enhanced System Modeling Language diagrams 

**Authors**: Matthew Anderson Hendricks, Alice Cicirello  

**Link**: [PDF](https://arxiv.org/pdf/2507.06803)  

**Abstract**: This paper contributes to speeding up the design and deployment of engineering dynamical systems by proposing a strategy for exploiting domain and expert knowledge for the automated generation of dynamical system computational model starting from a corpus of document relevant to the dynamical system of interest and an input document describing the specific system. This strategy is implemented in five steps and, crucially, it uses system modeling language diagrams (SysML) to extract accurate information about the dependencies, attributes, and operations of components. Natural Language Processing (NLP) strategies and Large Language Models (LLMs) are employed in specific tasks to improve intermediate outputs of the SySML diagrams automated generation, such as: list of key nouns; list of extracted relationships; list of key phrases and key relationships; block attribute values; block relationships; and BDD diagram generation. The applicability of automated SysML diagram generation is illustrated with different case studies. The computational models of complex dynamical systems from SysML diagrams are then obtained via code generation and computational model generation steps. In the code generation step, NLP strategies are used for summarization, while LLMs are used for validation only. The proposed approach is not limited to a specific system, domain, or computational software. The applicability of the proposed approach is shown via an end-to-end example from text to model of a simple pendulum, showing improved performance compared to results yielded by LLMs only. 

---
# On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks 

**Authors**: Stephen Obadinma, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06489)  

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses. 

---
# Exploring Task Performance with Interpretable Models via Sparse Auto-Encoders 

**Authors**: Shun Wang, Tyler Loakman, Youbo Lei, Yi Liu, Bohao Yang, Yuting Zhao, Dong Yang, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.06427)  

**Abstract**: Large Language Models (LLMs) are traditionally viewed as black-box algorithms, therefore reducing trustworthiness and obscuring potential approaches to increasing performance on downstream tasks. In this work, we apply an effective LLM decomposition method using a dictionary-learning approach with sparse autoencoders. This helps extract monosemantic features from polysemantic LLM neurons. Remarkably, our work identifies model-internal misunderstanding, allowing the automatic reformulation of the prompts with additional annotations to improve the interpretation by LLMs. Moreover, this approach demonstrates a significant performance improvement in downstream tasks, such as mathematical reasoning and metaphor detection. 

---
# Exploring LLMs for Predicting Tutor Strategy and Student Outcomes in Dialogues 

**Authors**: Fareya Ikram, Alexander Scarlatos, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06910)  

**Abstract**: Tutoring dialogues have gained significant attention in recent years, given the prominence of online learning and the emerging tutoring abilities of artificial intelligence (AI) agents powered by large language models (LLMs). Recent studies have shown that the strategies used by tutors can have significant effects on student outcomes, necessitating methods to predict how tutors will behave and how their actions impact students. However, few works have studied predicting tutor strategy in dialogues. Therefore, in this work we investigate the ability of modern LLMs, particularly Llama 3 and GPT-4o, to predict both future tutor moves and student outcomes in dialogues, using two math tutoring dialogue datasets. We find that even state-of-the-art LLMs struggle to predict future tutor strategy while tutor strategy is highly indicative of student outcomes, outlining a need for more powerful methods to approach this task. 

---
# UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations 

**Authors**: Fengran Mo, Yifan Gao, Chuan Meng, Xin Liu, Zhuofeng Wu, Kelong Mao, Zhengyang Wang, Pei Chen, Zheng Li, Xian Li, Bing Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07030)  

**Abstract**: The rapid advancement of conversational search systems revolutionizes how information is accessed by enabling the multi-turn interaction between the user and the system. Existing conversational search systems are usually built with two different models. This separation restricts the system from leveraging the intrinsic knowledge of the models simultaneously, which cannot ensure the effectiveness of retrieval benefiting the generation. The existing studies for developing unified models cannot fully address the aspects of understanding conversational context, managing retrieval independently, and generating responses. In this paper, we explore how to unify dense retrieval and response generation for large language models in conversation. We conduct joint fine-tuning with different objectives and design two mechanisms to reduce the inconsistency risks while mitigating data discrepancy. The evaluations on five conversational search datasets demonstrate that our unified model can mutually improve both tasks and outperform the existing baselines. 

---
# Reward Models Can Improve Themselves: Reward-Guided Adversarial Failure Mode Discovery for Robust Reward Modeling 

**Authors**: Pankayaraj Pathmanathan, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06419)  

**Abstract**: Reward modeling (RM), which captures human preferences to align large language models (LLMs), is increasingly employed in tasks such as model finetuning, response filtering, and ranking. However, due to the inherent complexity of human preferences and the limited coverage of available datasets, reward models often fail under distributional shifts or adversarial perturbations. Existing approaches for identifying such failure modes typically rely on prior knowledge about preference distributions or failure attributes, limiting their practicality in real-world settings where such information is unavailable. In this work, we propose a tractable, preference-distribution agnostic method for discovering reward model failure modes via reward guided controlled decoding. Building on this, we introduce REFORM, a self-improving reward modeling framework that enhances robustness by using the reward model itself to guide the generation of falsely scored responses. These adversarial examples are then used to augment the training data and patch the reward model's misaligned behavior. We evaluate REFORM on two widely used preference datasets Anthropic Helpful Harmless (HH) and PKU Beavertails and demonstrate that it significantly improves robustness without sacrificing reward quality. Notably, REFORM preserves performance both in direct evaluation and in downstream policy training, and further improves alignment quality by removing spurious correlations. 

---
# ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time 

**Authors**: Kiarash Zahirnia, Zahra Golpayegani, Walid Ahmad, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06313)  

**Abstract**: Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy. 

---
# Humans overrely on overconfident language models, across languages 

**Authors**: Neil Rathi, Dan Jurafsky, Kaitlyn Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06306)  

**Abstract**: As large language models (LLMs) are deployed globally, it is crucial that their responses are calibrated across languages to accurately convey uncertainty and limitations. Previous work has shown that LLMs are linguistically overconfident in English, leading users to overrely on confident generations. However, the usage and interpretation of epistemic markers (e.g., 'It's definitely,' 'I think') can differ sharply across languages. Here, we study the risks of multilingual linguistic (mis)calibration, overconfidence, and overreliance across five languages to evaluate the safety of LLMs in a global context.
We find that overreliance risks are high across all languages. We first analyze the distribution of LLM-generated epistemic markers, and observe that while LLMs are cross-linguistically overconfident, they are also sensitive to documented linguistic variation. For example, models generate the most markers of uncertainty in Japanese and the most markers of certainty in German and Mandarin. We then measure human reliance rates across languages, finding that while users strongly rely on confident LLM generations in all languages, reliance behaviors differ cross-linguistically: for example, users rely significantly more on expressions of uncertainty in Japanese than in English. Taken together, these results indicate high risk of reliance on overconfident model generations across languages. Our findings highlight the challenges of multilingual linguistic calibration and stress the importance of culturally and linguistically contextualized model safety evaluations. 

---
# Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities 

**Authors**: Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ilaï Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ramé, Sagar Waghmare, Helen Miller, Vaishakh Keshava, Ying Jian, Xiaofan Zhang, Raluca Ada Popa, Kedar Dhamdhere, Blaž Bratanič, Kyuyeun Kim, Terry Koo, Ferran Alet, Yi-ting Chen, Arsha Nagrani, Hannah Muckenhirn, Zhiyuan Zhang, Corbin Quick, Filip Pavetić, Duc Dung Nguyen, Joao Carreira, Michael Elabd, Haroon Qureshi, Fabian Mentzer, Yao-Yuan Yang, Danielle Eisenbud, Anmol Gulati, Ellie Talius, Eric Ni, Sahra Ghalebikesabi, Edouard Yvinec, Alaa Saade, Thatcher Ulrich, Lorenzo Blanco, Dan A. Calian, Muhuan Huang, Aäron van den Oord, Naman Goyal, Terry Chen, Praynaa Rawlani, Christian Schallhart, Swachhand Lokhande, Xianghong Luo, Jyn Shan, Ceslee Montgomery, Victoria Krakovna, Federico Piccinini, Omer Barak, Jingyu Cui, Yiling Jia, Mikhail Dektiarev, Alexey Kolganov, Shiyu Huang, Zhe Chen, Xingyu Wang, Jessica Austin, Peter de Boursac, Evgeny Sluzhaev, Frank Ding, Huijian Li, Surya Bhupatiraju  

**Link**: [PDF](https://arxiv.org/pdf/2507.06261)  

**Abstract**: In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving. 

---
# Expediting data extraction using a large language model (LLM) and scoping review protocol: a methodological study within a complex scoping review 

**Authors**: James Stewart-Evans, Emma Wilson, Tessa Langley, Andrew Prayle, Angela Hands, Karen Exley, Jo Leonardi-Bee  

**Link**: [PDF](https://arxiv.org/pdf/2507.06623)  

**Abstract**: The data extraction stages of reviews are resource-intensive, and researchers may seek to expediate data extraction using online (large language models) LLMs and review protocols. Claude 3.5 Sonnet was used to trial two approaches that used a review protocol to prompt data extraction from 10 evidence sources included in a case study scoping review. A protocol-based approach was also used to review extracted data. Limited performance evaluation was undertaken which found high accuracy for the two extraction approaches (83.3% and 100%) when extracting simple, well-defined citation details; accuracy was lower (9.6% and 15.8%) when extracting more complex, subjective data items. Considering all data items, both approaches had precision >90% but low recall (<25%) and F1 scores (<40%). The context of a complex scoping review, open response types and methodological approach likely impacted performance due to missed and misattributed data. LLM feedback considered the baseline extraction accurate and suggested minor amendments: four of 15 (26.7%) to citation details and 8 of 38 (21.1%) to key findings data items were considered to potentially add value. However, when repeating the process with a dataset featuring deliberate errors, only 2 of 39 (5%) errors were detected. Review-protocol-based methods used for expediency require more robust performance evaluation across a range of LLMs and review contexts with comparison to conventional prompt engineering approaches. We recommend researchers evaluate and report LLM performance if using them similarly to conduct data extraction or review extracted data. LLM feedback contributed to protocol adaptation and may assist future review protocol drafting. 

---
# Large Language Model for Extracting Complex Contract Information in Industrial Scenes 

**Authors**: Yunyang Cao, Yanjun Li, Silong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.06539)  

**Abstract**: This paper proposes a high-quality dataset construction method for complex contract information extraction tasks in industrial scenarios and fine-tunes a large language model based on this dataset. Firstly, cluster analysis is performed on industrial contract texts, and GPT-4 and GPT-3.5 are used to extract key information from the original contract data, obtaining high-quality data annotations. Secondly, data augmentation is achieved by constructing new texts, and GPT-3.5 generates unstructured contract texts from randomly combined keywords, improving model robustness. Finally, the large language model is fine-tuned based on the high-quality dataset. Experimental results show that the model achieves excellent overall performance while ensuring high field recall and precision and considering parsing efficiency. LoRA, data balancing, and data augmentation effectively enhance model accuracy and robustness. The proposed method provides a novel and efficient solution for industrial contract information extraction tasks. 

---
# Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs 

**Authors**: Yahan Yu, Yuyang Dong, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2507.06999)  

**Abstract**: Reasoning is a key capability for large language models (LLMs), particularly when applied to complex tasks such as mathematical problem solving. However, multimodal reasoning research still requires further exploration of modality alignment and training costs. Many of these approaches rely on additional data annotation and relevant rule-based rewards to enhance the understanding and reasoning ability, which significantly increases training costs and limits scalability. To address these challenges, we propose the Deliberate-to-Intuitive reasoning framework (D2I) that improves the understanding and reasoning ability of multimodal LLMs (MLLMs) without extra annotations and complex rewards. Specifically, our method sets deliberate reasoning strategies to enhance modality alignment only through the rule-based format reward during training. While evaluating, the reasoning style shifts to intuitive, which removes deliberate reasoning strategies during training and implicitly reflects the model's acquired abilities in the response. D2I outperforms baselines across both in-domain and out-of-domain benchmarks. Our findings highlight the role of format reward in fostering transferable reasoning skills in MLLMs, and inspire directions for decoupling training-time reasoning depth from test-time response flexibility. 

---
# Civil Society in the Loop: Feedback-Driven Adaptation of (L)LM-Assisted Classification in an Open-Source Telegram Monitoring Tool 

**Authors**: Milena Pustet, Elisabeth Steffen, Helena Mihaljević, Grischa Stanjek, Yannis Illies  

**Link**: [PDF](https://arxiv.org/pdf/2507.06734)  

**Abstract**: The role of civil society organizations (CSOs) in monitoring harmful online content is increasingly crucial, especially as platform providers reduce their investment in content moderation. AI tools can assist in detecting and monitoring harmful content at scale. However, few open-source tools offer seamless integration of AI models and social media monitoring infrastructures. Given their thematic expertise and contextual understanding of harmful content, CSOs should be active partners in co-developing technological tools, providing feedback, helping to improve models, and ensuring alignment with stakeholder needs and values, rather than as passive 'consumers'. However, collaborations between the open source community, academia, and civil society remain rare, and research on harmful content seldom translates into practical tools usable by civil society actors. This work in progress explores how CSOs can be meaningfully involved in an AI-assisted open-source monitoring tool of anti-democratic movements on Telegram, which we are currently developing in collaboration with CSO stakeholders. 

---
# Squeeze the Soaked Sponge: Efficient Off-policy Reinforcement Finetuning for Large Language Model 

**Authors**: Jing Liang, Hongyao Tang, Yi Ma, Jinyi Liu, Yan Zheng, Shuyue Hu, Lei Bai, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06892)  

**Abstract**: Reinforcement Learning (RL) has demonstrated its potential to improve the reasoning ability of Large Language Models (LLMs). One major limitation of most existing Reinforcement Finetuning (RFT) methods is that they are on-policy RL in nature, i.e., data generated during the past learning process is not fully utilized. This inevitably comes at a significant cost of compute and time, posing a stringent bottleneck on continuing economic and efficient scaling. To this end, we launch the renaissance of off-policy RL and propose Reincarnating Mix-policy Proximal Policy Gradient (ReMix), a general approach to enable on-policy RFT methods like PPO and GRPO to leverage off-policy data. ReMix consists of three major components: (1) Mix-policy proximal policy gradient with an increased Update-To-Data (UTD) ratio for efficient training; (2) KL-Convex policy constraint to balance the trade-off between stability and flexibility; (3) Policy reincarnation to achieve a seamless transition from efficient early-stage learning to steady asymptotic improvement. In our experiments, we train a series of ReMix models upon PPO, GRPO and 1.5B, 7B base models. ReMix shows an average Pass@1 accuracy of 52.10% (for 1.5B model) with 0.079M response rollouts, 350 training steps and achieves 63.27%/64.39% (for 7B model) with 0.007M/0.011M response rollouts, 50/75 training steps, on five math reasoning benchmarks (i.e., AIME'24, AMC'23, Minerva, OlympiadBench, and MATH500). Compared with 15 recent advanced models, ReMix shows SOTA-level performance with an over 30x to 450x reduction in training cost in terms of rollout data volume. In addition, we reveal insightful findings via multifaceted analysis, including the implicit preference for shorter responses due to the Whipping Effect of off-policy discrepancy, the collapse mode of self-reflection behavior under the presence of severe off-policyness, etc. 

---
# A Semantic Parsing Framework for End-to-End Time Normalization 

**Authors**: Xin Su, Sungduk Yu, Phillip Howard, Steven Bethard  

**Link**: [PDF](https://arxiv.org/pdf/2507.06450)  

**Abstract**: Time normalization is the task of converting natural language temporal expressions into machine-readable representations. It underpins many downstream applications in information retrieval, question answering, and clinical decision-making. Traditional systems based on the ISO-TimeML schema limit expressivity and struggle with complex constructs such as compositional, event-relative, and multi-span time expressions. In this work, we introduce a novel formulation of time normalization as a code generation task grounded in the SCATE framework, which defines temporal semantics through symbolic and compositional operators. We implement a fully executable SCATE Python library and demonstrate that large language models (LLMs) can generate executable SCATE code. Leveraging this capability, we develop an automatic data augmentation pipeline using LLMs to synthesize large-scale annotated data with code-level validation. Our experiments show that small, locally deployable models trained on this augmented data can achieve strong performance, outperforming even their LLM parents and enabling practical, accurate, and interpretable time normalization. 

---
# DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning 

**Authors**: Shreyas Vinaya Sathyanarayana, Rahil Shah, Sharanabasava D. Hiremath, Rishikesh Panda, Rahul Jana, Riya Singh, Rida Irfan, Ashwin Murali, Bharath Ramsundar  

**Link**: [PDF](https://arxiv.org/pdf/2507.07060)  

**Abstract**: Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses. 

---
# First Return, Entropy-Eliciting Explore 

**Authors**: Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, Qian Liu, Ge Zhang, Zejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.07017)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) improves the reasoning abilities of Large Language Models (LLMs) but it struggles with unstable exploration. We propose FR3E (First Return, Entropy-Eliciting Explore), a structured exploration framework that identifies high-uncertainty decision points in reasoning trajectories and performs targeted rollouts to construct semantically grounded intermediate feedback. Our method provides targeted guidance without relying on dense supervision. Empirical results on mathematical reasoning benchmarks(AIME24) show that FR3E promotes more stable training, produces longer and more coherent responses, and increases the proportion of fully correct trajectories. These results highlight the framework's effectiveness in improving LLM reasoning through more robust and structured exploration. 

---
# MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation 

**Authors**: Qilong Xing, Zikai Song, Youjia Zhang, Na Feng, Junqing Yu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06992)  

**Abstract**: Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation. 

---
# Developing and Maintaining an Open-Source Repository of AI Evaluations: Challenges and Insights 

**Authors**: Alexandra Abbas, Celia Waggoner, Justin Olive  

**Link**: [PDF](https://arxiv.org/pdf/2507.06893)  

**Abstract**: AI evaluations have become critical tools for assessing large language model capabilities and safety. This paper presents practical insights from eight months of maintaining $inspect\_evals$, an open-source repository of 70+ community-contributed AI evaluations. We identify key challenges in implementing and maintaining AI evaluations and develop solutions including: (1) a structured cohort management framework for scaling community contributions, (2) statistical methodologies for optimal resampling and cross-model comparison with uncertainty quantification, and (3) systematic quality control processes for reproducibility. Our analysis reveals that AI evaluation requires specialized infrastructure, statistical rigor, and community coordination beyond traditional software development practices. 

---
# The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover 

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06850)  

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors. 

---
# From Data-Centric to Sample-Centric: Enhancing LLM Reasoning via Progressive Optimization 

**Authors**: Xinjie Chen, Minpeng Liao, Guoxin Chen, Chengxi Li, Biao Fu, Kai Fan, Xinggao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06573)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has recently advanced the reasoning capabilities of large language models (LLMs). While prior work has emphasized algorithmic design, data curation, and reward shaping, we investigate RLVR from a sample-centric perspective and introduce LPPO (Learning-Progress and Prefix-guided Optimization), a framework of progressive optimization techniques. Our work addresses a critical question: how to best leverage a small set of trusted, high-quality demonstrations, rather than simply scaling up data volume. First, motivated by how hints aid human problem-solving, we propose prefix-guided sampling, an online data augmentation method that incorporates partial solution prefixes from expert demonstrations to guide the policy, particularly for challenging instances. Second, inspired by how humans focus on important questions aligned with their current capabilities, we introduce learning-progress weighting, a dynamic strategy that adjusts each training sample's influence based on model progression. We estimate sample-level learning progress via an exponential moving average of per-sample pass rates, promoting samples that foster learning and de-emphasizing stagnant ones. Experiments on mathematical-reasoning benchmarks demonstrate that our methods outperform strong baselines, yielding faster convergence and a higher performance ceiling. 

---
# Gradientsys: A Multi-Agent LLM Scheduler with ReAct Orchestration 

**Authors**: Xinyuan Song, Zeyu Wang, Siyi Wu, Tianyu Shi, Lynn Ai  

**Link**: [PDF](https://arxiv.org/pdf/2507.06520)  

**Abstract**: We present Gradientsys, a next-generation multi-agent scheduling framework that coordinates diverse specialized AI agents using a typed Model-Context Protocol (MCP) and a ReAct-based dynamic planning loop. At its core, Gradientsys employs an LLM-powered scheduler for intelligent one-to-many task dispatch, enabling parallel execution of heterogeneous agents such as PDF parsers, web search modules, GUI controllers, and web builders. The framework supports hybrid synchronous/asynchronous execution, respects agent capacity constraints, and incorporates a robust retry-and-replan mechanism to handle failures gracefully. To promote transparency and trust, Gradientsys includes an observability layer streaming real-time agent activity and intermediate reasoning via Server-Sent Events (SSE). We offer an architectural overview and evaluate Gradientsys against existing frameworks in terms of extensibility, scheduling topology, tool reusability, parallelism, and observability. Experiments on the GAIA general-assistant benchmark show that Gradientsys achieves higher task success rates with reduced latency and lower API costs compared to a MinionS-style baseline, demonstrating the strength of its LLM-driven multi-agent orchestration. 

---
# Towards LLM-based Root Cause Analysis of Hardware Design Failures 

**Authors**: Siyu Qiu, Muzhi Wang, Raheel Afsharmazayejani, Mohammad Moradi Shahmiri, Benjamin Tan, Hammond Pearce  

**Link**: [PDF](https://arxiv.org/pdf/2507.06512)  

**Abstract**: With advances in large language models (LLMs), new opportunities have emerged to develop tools that support the digital hardware design process. In this work, we explore how LLMs can assist with explaining the root cause of design issues and bugs that are revealed during synthesis and simulation, a necessary milestone on the pathway towards widespread use of LLMs in the hardware design process and for hardware security analysis. We find promising results: for our corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model reached a correct determination 100% of the time under pass@5 scoring, with other state of the art models and configurations usually achieving more than 80% performance and more than 90% when assisted with retrieval-augmented generation. 

---
# Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks 

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06274)  

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings. 

---
# Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World 

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06256)  

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures. 

---
# Towards Solving More Challenging IMO Problems via Decoupled Reasoning and Proving 

**Authors**: Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06804)  

**Abstract**: Automated Theorem Proving (ATP) in formal languages is a foundational challenge for AI. While Large Language Models (LLMs) have driven remarkable progress, a significant gap remains between their powerful informal reasoning capabilities and their weak formal proving performance. Recent studies show that the informal accuracy exceeds 80% while formal success remains below 8% on benchmarks like PutnamBench. We argue this gap persists because current state-of-the-art provers, by tightly coupling reasoning and proving, are trained with paradigms that inadvertently punish deep reasoning in favor of shallow, tactic-based strategies. To bridge this fundamental gap, we propose a novel framework that decouples high-level reasoning from low-level proof generation. Our approach utilizes two distinct, specialized models: a powerful, general-purpose Reasoner to generate diverse, strategic subgoal lemmas, and an efficient Prover to rigorously verify them. This modular design liberates the model's full reasoning potential and bypasses the pitfalls of end-to-end training. We evaluate our method on a challenging set of post-2000 IMO problems, a problem set on which no prior open-source prover has reported success. Our decoupled framework successfully solves 5 of these problems, demonstrating a significant step towards automated reasoning on exceptionally difficult mathematical challenges. To foster future research, we release our full dataset of generated and verified lemmas for a wide range of IMO problems, available at this https URL . 

---
# SkyVLN: Vision-and-Language Navigation and NMPC Control for UAVs in Urban Environments 

**Authors**: Tianshun Li, Tianyi Huai, Zhen Li, Yichun Gao, Haoang Li, Xinhu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06564)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have emerged as versatile tools across various sectors, driven by their mobility and adaptability. This paper introduces SkyVLN, a novel framework integrating vision-and-language navigation (VLN) with Nonlinear Model Predictive Control (NMPC) to enhance UAV autonomy in complex urban environments. Unlike traditional navigation methods, SkyVLN leverages Large Language Models (LLMs) to interpret natural language instructions and visual observations, enabling UAVs to navigate through dynamic 3D spaces with improved accuracy and robustness. We present a multimodal navigation agent equipped with a fine-grained spatial verbalizer and a history path memory mechanism. These components allow the UAV to disambiguate spatial contexts, handle ambiguous instructions, and backtrack when necessary. The framework also incorporates an NMPC module for dynamic obstacle avoidance, ensuring precise trajectory tracking and collision prevention. To validate our approach, we developed a high-fidelity 3D urban simulation environment using AirSim, featuring realistic imagery and dynamic urban elements. Extensive experiments demonstrate that SkyVLN significantly improves navigation success rates and efficiency, particularly in new and unseen environments. 

---
# Too Human to Model:The Uncanny Valley of LLMs in Social Simulation -- When Generative Language Agents Misalign with Modelling Principles 

**Authors**: Yongchao Zeng, Calum Brown, Mark Rounsevell  

**Link**: [PDF](https://arxiv.org/pdf/2507.06310)  

**Abstract**: Large language models (LLMs) have been increasingly used to build agents in social simulation because of their impressive abilities to generate fluent, contextually coherent dialogues. Such abilities can enhance the realism of models. However, the pursuit of realism is not necessarily compatible with the epistemic foundation of modelling. We argue that LLM agents, in many regards, are too human to model: they are too expressive, detailed and intractable to be consistent with the abstraction, simplification, and interpretability typically demanded by modelling. Through a model-building thought experiment that converts the Bass diffusion model to an LLM-based variant, we uncover five core dilemmas: a temporal resolution mismatch between natural conversation and abstract time steps; the need for intervention in conversations while avoiding undermining spontaneous agent outputs; the temptation to introduce rule-like instructions in prompts while maintaining conversational naturalness; the tension between role consistency and role evolution across time; and the challenge of understanding emergence, where system-level patterns become obscured by verbose micro textual outputs. These dilemmas steer the LLM agents towards an uncanny valley: not abstract enough to clarify underlying social mechanisms, while not natural enough to represent realistic human behaviour. This exposes an important paradox: the realism of LLM agents can obscure, rather than clarify, social dynamics when misapplied. We tease out the conditions in which LLM agents are ideally suited: where system-level emergence is not the focus, linguistic nuances and meaning are central, interactions unfold in natural time, and stable role identity is more important than long-term behavioural evolution. We call for repositioning LLM agents in the ecosystem of social simulation for future applications. 

---
