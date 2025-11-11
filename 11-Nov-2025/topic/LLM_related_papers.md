# Can LLM Annotations Replace User Clicks for Learning to Rank? 

**Authors**: Lulu Yu, Keping Bi, Jiafeng Guo, Shihao Liu, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06635)  

**Abstract**: Large-scale supervised data is essential for training modern ranking models, but obtaining high-quality human annotations is costly. Click data has been widely used as a low-cost alternative, and with recent advances in large language models (LLMs), LLM-based relevance annotation has emerged as another promising annotation. This paper investigates whether LLM annotations can replace click data for learning to rank (LTR) by conducting a comprehensive comparison across multiple dimensions. Experiments on both a public dataset, TianGong-ST, and an industrial dataset, Baidu-Click, show that click-supervised models perform better on high-frequency queries, while LLM annotation-supervised models are more effective on medium- and low-frequency queries. Further analysis shows that click-supervised models are better at capturing document-level signals such as authority or quality, while LLM annotation-supervised models are more effective at modeling semantic matching between queries and documents and at distinguishing relevant from non-relevant documents. Motivated by these observations, we explore two training strategies -- data scheduling and frequency-aware multi-objective learning -- that integrate both supervision signals. Both approaches enhance ranking performance across queries at all frequency levels, with the latter being more effective. Our code is available at this https URL. 

---
# TOOL4POI: A Tool-Augmented LLM Framework for Next POI Recommendation 

**Authors**: Dongsheng Wang, Shen Gao, Chengrui Huang, Yuxi Huang, Ruixiang Feng, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06405)  

**Abstract**: Next Point-of-Interest (POI) recommendation is a fundamental task in location-based services. While recent advances leverage Large Language Model (LLM) for sequential modeling, existing LLM-based approaches face two key limitations: (i) strong reliance on the contextual completeness of user histories, resulting in poor performance on out-of-history (OOH) scenarios; (ii) limited scalability, due to the restricted context window of LLMs, which limits their ability to access and process a large number of candidate POIs. To address these challenges, we propose Tool4POI, a novel tool-augmented framework that enables LLMs to perform open-set POI recommendation through external retrieval and reasoning. Tool4POI consists of three key modules: preference extraction module, multi-turn candidate retrieval module, and reranking module, which together summarize long-term user interests, interact with external tools to retrieve relevant POIs, and refine final recommendations based on recent behaviors. Unlike existing methods, Tool4POI requires no task-specific fine-tuning and is compatible with off-the-shelf LLMs in a plug-and-play manner. Extensive experiments on three real-world datasets show that Tool4POI substantially outperforms state-of-the-art baselines, achieving up to 40% accuracy on challenging OOH scenarios where existing methods fail, and delivering average improvements of 20% and 30% on Acc@5 and Acc@10, respectively. 

---
# When Evidence Contradicts: Toward Safer Retrieval-Augmented Generation in Healthcare 

**Authors**: Saeedeh Javadi, Sara Mirabi, Manan Gangar, Bahadorreza Ofoghi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06668)  

**Abstract**: In high-stakes information domains such as healthcare, where large language models (LLMs) can produce hallucinations or misinformation, retrieval-augmented generation (RAG) has been proposed as a mitigation strategy, grounding model outputs in external, domain-specific documents. Yet, this approach can introduce errors when source documents contain outdated or contradictory information. This work investigates the performance of five LLMs in generating RAG-based responses to medicine-related queries. Our contributions are three-fold: i) the creation of a benchmark dataset using consumer medicine information documents from the Australian Therapeutic Goods Administration (TGA), where headings are repurposed as natural language questions, ii) the retrieval of PubMed abstracts using TGA headings, stratified across multiple publication years, to enable controlled temporal evaluation of outdated evidence, and iii) a comparative analysis of the frequency and impact of outdated or contradictory content on model-generated responses, assessing how LLMs integrate and reconcile temporally inconsistent information. Our findings show that contradictions between highly similar abstracts do, in fact, degrade performance, leading to inconsistencies and reduced factual accuracy in model answers. These results highlight that retrieval similarity alone is insufficient for reliable medical RAG and underscore the need for contradiction-aware filtering strategies to ensure trustworthy responses in high-stakes domains. 

---
# Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models 

**Authors**: Tianrui Song, Wen-Shuo Chao, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07295)  

**Abstract**: Implicit feedback, employed in training recommender systems, unavoidably confronts noise due to factors such as misclicks and position bias. Previous studies have attempted to identify noisy samples through their diverged data patterns, such as higher loss values, and mitigate their influence through sample dropping or reweighting. However, we observed that noisy samples and hard samples display similar patterns, leading to hard-noisy confusion issue. Such confusion is problematic as hard samples are vital for modeling user preferences. To solve this problem, we propose LLMHNI framework, leveraging two auxiliary user-item relevance signals generated by Large Language Models (LLMs) to differentiate hard and noisy samples. LLMHNI obtains user-item semantic relevance from LLM-encoded embeddings, which is used in negative sampling to select hard negatives while filtering out noisy false negatives. An objective alignment strategy is proposed to project LLM-encoded embeddings, originally for general language tasks, into a representation space optimized for user-item relevance modeling. LLMHNI also exploits LLM-inferred logical relevance within user-item interactions to identify hard and noisy samples. These LLM-inferred interactions are integrated into the interaction graph and guide denoising with cross-graph contrastive alignment. To eliminate the impact of unreliable interactions induced by LLM hallucination, we propose a graph contrastive learning strategy that aligns representations from randomly edge-dropped views to suppress unreliable edges. Empirical results demonstrate that LLMHNI significantly improves denoising and recommendation performance. 

---
# Biomedical Hypothesis Explainability with Graph-Based Context Retrieval 

**Authors**: Ilya Tyagin, Saeideh Valipour, Aliaksandra Sikirzhytskaya, Michael Shtutman, Ilya Safro  

**Link**: [PDF](https://arxiv.org/pdf/2511.05498)  

**Abstract**: We introduce an explainability method for biomedical hypothesis generation systems, built on top of the novel Hypothesis Generation Context Retriever framework. Our approach combines semantic graph-based retrieval and relevant data-restrictive training to simulate real-world discovery constraints. Integrated with large language models (LLMs) via retrieval-augmented generation, the system explains hypotheses with contextual evidence using published scientific literature. We also propose a novel feedback loop approach, which iteratively identifies and corrects flawed parts of LLM-generated explanations, refining both the evidence paths and supporting context. We demonstrate the performance of our method with multiple large language models and evaluate the explanation and context retrieval quality through both expert-curated assessment and large-scale automated analysis. Our code is available at: this https URL. 

---
# Ontology Learning and Knowledge Graph Construction: A Comparison of Approaches and Their Impact on RAG Performance 

**Authors**: Tiago da Cruz, Bernardo Tavares, Francisco Belo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05991)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems combine Large Language Models (LLMs) with external knowledge, and their performance depends heavily on how that knowledge is represented. This study investigates how different Knowledge Graph (KG) construction strategies influence RAG performance. We compare a variety of approaches: standard vector-based RAG, GraphRAG, and retrieval over KGs built from ontologies derived either from relational databases or textual corpora. Results show that ontology-guided KGs incorporating chunk information achieve competitive performance with state-of-the-art frameworks, substantially outperforming vector retrieval baselines. Moreover, the findings reveal that ontology-guided KGs built from relational databases perform competitively to ones built with ontologies extracted from text, with the benefit of offering a dual advantage: they require a one-time-only ontology learning process, substantially reducing LLM usage costs; and avoid the complexity of ontology merging inherent to text-based approaches. 

---
# Retrieval Quality at Context Limit 

**Authors**: Max McKinnon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05850)  

**Abstract**: The ability of large language models (LLMs) to recall and retrieve information from long contexts is critical for many real-world applications. Prior work (Liu et al., 2023) reported that LLMs suffer significant drops in retrieval accuracy for facts placed in the middle of large contexts, an effect known as "Lost in the Middle" (LITM). We find the model Gemini 2.5 Flash can answer needle-in-a-haystack questions with great accuracy regardless of document position including when the document is nearly at the input context limit. Our results suggest that the "Lost in the Middle" effect is not present for simple factoid Q\&A in Gemini 2.5 Flash, indicating substantial improvements in long-context retrieval. 

---
# Customized Retrieval-Augmented Generation with LLM for Debiasing Recommendation Unlearning 

**Authors**: Haichao Zhang, Chong Zhang, Peiyu Hu, Shi Qiu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05494)  

**Abstract**: Modern recommender systems face a critical challenge in complying with privacy regulations like the 'right to be forgotten': removing a user's data without disrupting recommendations for others. Traditional unlearning methods address this by partial model updates, but introduce propagation bias--where unlearning one user's data distorts recommendations for behaviorally similar users, degrading system accuracy. While retraining eliminates bias, it is computationally prohibitive for large-scale systems. To address this challenge, we propose CRAGRU, a novel framework leveraging Retrieval-Augmented Generation (RAG) for efficient, user-specific unlearning that mitigates bias while preserving recommendation quality. CRAGRU decouples unlearning into distinct retrieval and generation stages. In retrieval, we employ three tailored strategies designed to precisely isolate the target user's data influence, minimizing collateral impact on unrelated users and enhancing unlearning efficiency. Subsequently, the generation stage utilizes an LLM, augmented with user profiles integrated into prompts, to reconstruct accurate and personalized recommendations without needing to retrain the entire base model. Experiments on three public datasets demonstrate that CRAGRU effectively unlearns targeted user data, significantly mitigating unlearning bias by preventing adverse impacts on non-target users, while maintaining recommendation performance comparable to fully trained original models. Our work highlights the promise of RAG-based architectures for building robust and privacy-preserving recommender systems. The source code is available at: this https URL. 

---
# DOCUEVAL: An LLM-based AI Engineering Tool for Building Customisable Document Evaluation Workflows 

**Authors**: Hao Zhang, Qinghua Lu, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05496)  

**Abstract**: Foundation models, such as large language models (LLMs), have the potential to streamline evaluation workflows and improve their performance. However, practical adoption faces challenges, such as customisability, accuracy, and scalability. In this paper, we present DOCUEVAL, an AI engineering tool for building customisable DOCUment EVALuation workflows. DOCUEVAL supports advanced document processing and customisable workflow design which allow users to define theory-grounded reviewer roles, specify evaluation criteria, experiment with different reasoning strategies and choose the assessment style. To ensure traceability, DOCUEVAL provides comprehensive logging of every run, along with source attribution and configuration management, allowing systematic comparison of results across alternative setups. By integrating these capabilities, DOCUEVAL directly addresses core software engineering challenges, including how to determine whether evaluators are "good enough" for deployment and how to empirically compare different evaluation strategies. We demonstrate the usefulness of DOCUEVAL through a real-world academic peer review case, showing how DOCUEVAL enables both the engineering of evaluators and scalable, reliable document evaluation. 

---
# AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs 

**Authors**: Yubo Wang, Haoyang Li, Fei Teng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05549)  

**Abstract**: Graph-based retrieval-augmented generation (Graph-based RAG) has demonstrated significant potential in enhancing Large Language Models (LLMs) with structured knowledge. However, existing methods face three critical challenges: Inaccurate Graph Construction, caused by LLM hallucination; Poor Reasoning Ability, caused by failing to generate explicit reasons telling LLM why certain chunks were selected; and Inadequate Answering, which only partially answers the query due to the inadequate LLM reasoning, making their performance lag behind NaiveRAG on certain tasks. To address these issues, we propose AGRAG, an advanced graph-based retrieval-augmented generation framework. When constructing the graph, AGRAG substitutes the widely used LLM entity extraction method with a statistics-based method, avoiding hallucination and error propagation. When retrieval, AGRAG formulates the graph reasoning procedure as the Minimum Cost Maximum Influence (MCMI) subgraph generation problem, where we try to include more nodes with high influence score, but with less involving edge cost, to make the generated reasoning paths more comprehensive. We prove this problem to be NP-hard, and propose a greedy algorithm to solve it. The MCMI subgraph generated can serve as explicit reasoning paths to tell LLM why certain chunks were retrieved, thereby making the LLM better focus on the query-related part contents of the chunks, reducing the impact of noise, and improving AGRAG's reasoning ability. Furthermore, compared with the simple tree-structured reasoning paths, our MCMI subgraph can allow more complex graph structures, such as cycles, and improve the comprehensiveness of the generated reasoning paths. 

---
# Evaluating Online Moderation Via LLM-Powered Counterfactual Simulations 

**Authors**: Giacomo Fidone, Lucia Passaro, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2511.07204)  

**Abstract**: Online Social Networks (OSNs) widely adopt content moderation to mitigate the spread of abusive and toxic discourse. Nonetheless, the real effectiveness of moderation interventions remains unclear due to the high cost of data collection and limited experimental control. The latest developments in Natural Language Processing pave the way for a new evaluation approach. Large Language Models (LLMs) can be successfully leveraged to enhance Agent-Based Modeling and simulate human-like social behavior with unprecedented degree of believability. Yet, existing tools do not support simulation-based evaluation of moderation strategies. We fill this gap by designing a LLM-powered simulator of OSN conversations enabling a parallel, counterfactual simulation where toxic behavior is influenced by moderation interventions, keeping all else equal. We conduct extensive experiments, unveiling the psychological realism of OSN agents, the emergence of social contagion phenomena and the superior effectiveness of personalized moderation strategies. 

---
# DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas 

**Authors**: Zhen Wang, Yufan Zhou, Zhongyan Luo, Lyumanshan Ye, Adam Wood, Man Yao, Luoshang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07338)  

**Abstract**: Simulating human profiles by instilling personas into large language models (LLMs) is rapidly transforming research in agentic behavioral simulation, LLM personalization, and human-AI alignment. However, most existing synthetic personas remain shallow and simplistic, capturing minimal attributes and failing to reflect the rich complexity and diversity of real human identities. We introduce DEEPPERSONA, a scalable generative engine for synthesizing narrative-complete synthetic personas through a two-stage, taxonomy-guided method. First, we algorithmically construct the largest-ever human-attribute taxonomy, comprising over hundreds of hierarchically organized attributes, by mining thousands of real user-ChatGPT conversations. Second, we progressively sample attributes from this taxonomy, conditionally generating coherent and realistic personas that average hundreds of structured attributes and roughly 1 MB of narrative text, two orders of magnitude deeper than prior works. Intrinsic evaluations confirm significant improvements in attribute diversity (32 percent higher coverage) and profile uniqueness (44 percent greater) compared to state-of-the-art baselines. Extrinsically, our personas enhance GPT-4.1-mini's personalized question answering accuracy by 11.6 percent on average across ten metrics and substantially narrow (by 31.7 percent) the gap between simulated LLM citizens and authentic human responses in social surveys. Our generated national citizens reduced the performance gap on the Big Five personality test by 17 percent relative to LLM-simulated citizens. DEEPPERSONA thus provides a rigorous, scalable, and privacy-free platform for high-fidelity human simulation and personalized AI research. 

---
# MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks 

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Guangze Ye, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2511.07107)  

**Abstract**: Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment. 

---
# Increasing AI Explainability by LLM Driven Standard Processes 

**Authors**: Marc Jansen, Marcel Pehlke  

**Link**: [PDF](https://arxiv.org/pdf/2511.07083)  

**Abstract**: This paper introduces an approach to increasing the explainability of artificial intelligence (AI) systems by embedding Large Language Models (LLMs) within standardized analytical processes. While traditional explainable AI (XAI) methods focus on feature attribution or post-hoc interpretation, the proposed framework integrates LLMs into defined decision models such as Question-Option-Criteria (QOC), Sensitivity Analysis, Game Theory, and Risk Management. By situating LLM reasoning within these formal structures, the approach transforms opaque inference into transparent and auditable decision traces. A layered architecture is presented that separates the reasoning space of the LLM from the explainable process space above it. Empirical evaluations show that the system can reproduce human-level decision logic in decentralized governance, systems analysis, and strategic reasoning contexts. The results suggest that LLM-driven standard processes provide a foundation for reliable, interpretable, and verifiable AI-supported decision making. 

---
# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07061)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets-- IEMOCAP and MELD --show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding. 

---
# MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning 

**Authors**: Jinhao Chen, Zhen Yang, Jianxin Shi, Tianyu Wo, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06805)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in vision-language answering tasks. Despite their strengths, these models often encounter challenges in achieving complex reasoning tasks such as mathematical problem-solving. Previous works have focused on fine-tuning on specialized mathematical datasets. However, these datasets are typically distilled directly from teacher models, which capture only static reasoning patterns and leaving substantial gaps compared to student models. This reliance on fixed teacher-derived datasets not only restricts the model's ability to adapt to novel or more intricate questions that extend beyond the confines of the training data, but also lacks the iterative depth needed for robust generalization. To overcome these limitations, we propose \textbf{\method}, a \textbf{Math}ematical \textbf{S}elf-\textbf{E}volving framework for MLLMs. In contrast to traditional one-shot fine-tuning paradigms, \method iteratively refines the model through cycles of inference, reflection, and reward-based feedback. Specifically, we leverage iterative fine-tuning by incorporating correct reasoning paths derived from previous-stage inference and integrating reflections from a specialized Outcome Reward Model (ORM). To verify the effectiveness of \method, we evaluate it on a suite of challenging benchmarks, demonstrating significant performance gains over backbone models. Notably, our experimental results on MathVL-test surpass the leading open-source multimodal mathematical reasoning model QVQ. Our code and models are available at \texttt{https://zheny2751\this http URL\allowbreak this http URL}. 

---
# GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization 

**Authors**: Moriya Dechtiar, Daniel Martin Katz, Mari Sundaresan, Sylvain Jaume, Hongming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06618)  

**Abstract**: Contracts are complex documents featuring detailed formal structures, explicit and implicit dependencies and rich semantic content. Given these document properties, contract drafting and manual examination of contracts have proven to be both arduous and susceptible to errors. This work aims to simplify and automate the task of contract review and analysis using a novel framework for transforming legal contracts into structured semantic graphs, enabling computational analysis and data-driven insights. We introduce a detailed ontology mapping core legal contract elements to their graph-theoretic equivalents of nodes and edges. We then present a reinforcement learning based Large Language Model (LLM) framework for segmentation and extraction of entities and relationships from contracts. Our method, GRAPH-GRPO-LEX, incorporates both LLMs and reinforcement learning with group relative policy optimization (GRPO). By applying a carefully drafted reward function of graph metrics, we demonstrate the ability to automatically identify direct relationships between clauses, and even uncover hidden dependencies. Our introduction of the gated GRPO approach shows a strong learning signal and can move contract analysis from a linear, manual reading process to an easily visualized graph. This allows for a more dynamic analysis, including building the groundwork for contract linting similar to what is now practiced in software engineering. 

---
# Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision 

**Authors**: Yimei Zhang, Guojiang Shen, Kaili Ning, Tongwei Ren, Xuebo Qiu, Mengmeng Wang, Xiangjie Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.07062)  

**Abstract**: Region representation learning plays a pivotal role in urban computing by extracting meaningful features from unlabeled urban data. Analogous to how perceived facial age reflects an individual's health, the visual appearance of a city serves as its ``portrait", encapsulating latent socio-economic and environmental characteristics. Recent studies have explored leveraging Large Language Models (LLMs) to incorporate textual knowledge into imagery-based urban region representation learning. However, two major challenges remain: i)~difficulty in aligning fine-grained visual features with long captions, and ii) suboptimal knowledge incorporation due to noise in LLM-generated captions. To address these issues, we propose a novel pre-training framework called UrbanLN that improves Urban region representation learning through Long-text awareness and Noise suppression. Specifically, we introduce an information-preserved stretching interpolation strategy that aligns long captions with fine-grained visual semantics in complex urban scenes. To effectively mine knowledge from LLM-generated captions and filter out noise, we propose a dual-level optimization strategy. At the data level, a multi-model collaboration pipeline automatically generates diverse and reliable captions without human intervention. At the model level, we employ a momentum-based self-distillation mechanism to generate stable pseudo-targets, facilitating robust cross-modal learning under noisy conditions. Extensive experiments across four real-world cities and various downstream tasks demonstrate the superior performance of our UrbanLN. 

---
# Optimizing Chain-of-Thought Confidence via Topological and Dirichlet Risk Analysis 

**Authors**: Abhishek More, Anthony Zhang, Nicole Bonilla, Ashvik Vivekan, Kevin Zhu, Parham Sharafoleslami, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.06437)  

**Abstract**: Chain-of-thought (CoT) prompting enables Large Language Models to solve complex problems, but deploying these models safely requires reliable confidence estimates, a capability where existing methods suffer from poor calibration and severe overconfidence on incorrect predictions. We propose Enhanced Dirichlet and Topology Risk (EDTR), a novel decoding strategy that combines topological analysis with Dirichlet-based uncertainty quantification to measure LLM confidence across multiple reasoning paths. EDTR treats each CoT as a vector in high-dimensional space and extracts eight topological risk features capturing the geometric structure of reasoning distributions: tighter, more coherent clusters indicate higher confidence while dispersed, inconsistent paths signal uncertainty. We evaluate EDTR against three state-of-the-art calibration methods across four diverse reasoning benchmarks spanning olympiad-level mathematics (AIME), grade school math (GSM8K), commonsense reasoning, and stock price prediction \cite{zhang2025aime, cobbe2021training, talmor-etal-2019-commonsenseqa, yahoo_finance}. EDTR achieves 41\% better calibration than competing methods with an average ECE of 0.287 and the best overall composite score of 0.672, while notably achieving perfect accuracy on AIME and exceptional calibration on GSM8K with an ECE of 0.107, domains where baselines exhibit severe overconfidence. Our work provides a geometric framework for understanding and quantifying uncertainty in multi-step LLM reasoning, enabling more reliable deployment where calibrated confidence estimates are essential. 

---
# Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives 

**Authors**: Chloe Li, Mary Phuong, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06626)  

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to admit their factual mistakes when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AI systems. 

---
# SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.06411)  

**Abstract**: The soft-thinking paradigm for Large Language Model (LLM) reasoning can outperform the conventional discrete-token Chain-of-Thought (CoT) reasoning in some scenarios, underscoring its research and application value. However, while the discrete-token CoT reasoning pattern can be reinforced through policy optimization algorithms such as group relative policy optimization (GRPO), extending the soft-thinking pattern with Reinforcement Learning (RL) remains challenging. This difficulty stems from the complexities of injecting stochasticity into soft-thinking tokens and updating soft-thinking policies accordingly. As a result, previous attempts to combine soft-thinking with GRPO typically underperform their discrete-token GRPO counterparts. To fully unlock the potential of soft-thinking, this paper presents a novel policy optimization algorithm, SofT-GRPO, to reinforce LLMs under the soft-thinking reasoning pattern. SofT-GRPO injects the Gumbel noise into logits, employs the Gumbel-Softmax technique to avoid soft-thinking tokens outside the pre-trained embedding space, and leverages the reparameterization trick in policy gradient. We conduct experiments across base LLMs ranging from 1.5B to 7B parameters, and results demonstrate that SofT-GRPO enables soft-thinking LLMs to slightly outperform discrete-token GRPO on Pass@1 (+0.13% on average accuracy), while exhibiting a substantial uplift on Pass@32 (+2.19% on average accuracy). Codes and weights are available on this https URL 

---
# AUTO-Explorer: Automated Data Collection for GUI Agent 

**Authors**: Xiangwu Guo, Difei Gao, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06417)  

**Abstract**: Recent advancements in GUI agents have significantly expanded their ability to interpret natural language commands to manage software interfaces. However, acquiring GUI data remains a significant challenge. Existing methods often involve designing automated agents that browse URLs from the Common Crawl, using webpage HTML to collect screenshots and corresponding annotations, including the names and bounding boxes of UI elements. However, this method is difficult to apply to desktop software or some newly launched websites not included in the Common Crawl. While we expect the model to possess strong generalization capabilities to handle this, it is still crucial for personalized scenarios that require rapid and perfect adaptation to new software or websites. To address this, we propose an automated data collection method with minimal annotation costs, named Auto-Explorer. It incorporates a simple yet effective exploration mechanism that autonomously parses and explores GUI environments, gathering data efficiently. Additionally, to assess the quality of exploration, we have developed the UIXplore benchmark. This benchmark creates environments for explorer agents to discover and save software states. Using the data gathered, we fine-tune a multimodal large language model (MLLM) and establish a GUI element grounding testing set to evaluate the effectiveness of the exploration strategies. Our experiments demonstrate the superior performance of Auto-Explorer, showing that our method can quickly enhance the capabilities of an MLLM in explored software. 

---
# LLM Driven Processes to Foster Explainable AI 

**Authors**: Marcel Pehlke, Marc Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07086)  

**Abstract**: We present a modular, explainable LLM-agent pipeline for decision support that externalizes reasoning into auditable artifacts. The system instantiates three frameworks: Vester's Sensitivity Model (factor set, signed impact matrix, systemic roles, feedback loops); normal-form games (strategies, payoff matrix, equilibria); and sequential games (role-conditioned agents, tree construction, backward induction), with swappable modules at every step. LLM components (default: GPT-5) are paired with deterministic analyzers for equilibria and matrix-based role classification, yielding traceable intermediates rather than opaque outputs. In a real-world logistics case (100 runs), mean factor alignment with a human baseline was 55.5\% over 26 factors and 62.9\% on the transport-core subset; role agreement over matches was 57\%. An LLM judge using an eight-criterion rubric (max 100) scored runs on par with a reconstructed human baseline. Configurable LLM pipelines can thus mimic expert workflows with transparent, inspectable steps. 

---
# Efficient LLM Safety Evaluation through Multi-Agent Debate 

**Authors**: Dachuan Lin, Guobin Shen, Zihao Yang, Tianrong Liu, Dongcheng Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06396)  

**Abstract**: Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-Judge frameworks, but the high cost of frontier models limits scalability. We propose a cost-efficient multi-agent judging framework that employs Small Language Models (SLMs) through structured debates among critic, defender, and judge agents. To rigorously assess safety judgments, we construct HAJailBench, a large-scale human-annotated jailbreak benchmark comprising 12,000 adversarial interactions across diverse attack methods and target models. The dataset provides fine-grained, expert-labeled ground truth for evaluating both safety robustness and judge reliability. Our SLM-based framework achieves agreement comparable to GPT-4o judges on HAJailBench while substantially reducing inference cost. Ablation results show that three rounds of debate yield the optimal balance between accuracy and efficiency. These findings demonstrate that structured, value-aligned debate enables SLMs to capture semantic nuances of jailbreak attacks and that HAJailBench offers a reliable foundation for scalable LLM safety evaluation. 

---
# RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services 

**Authors**: Fei Zhao, Chonggang Lu, Haofu Qian, Fangcheng Shi, Zijie Meng, Jianzhao Huang, Xu Tang, Zheyong Xie, Zheyu Ye, Zhe Xu, Yao Hu, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07070)  

**Abstract**: As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervised fine-tuning (SFT) can specialize models but often triggers a ``seesaw'' between in-distribution gains and out-of-distribution robustness, especially for smaller models. To address these challenges, we introduce RedOne 2.0, an SNS-oriented LLM trained with a progressive, RL-prioritized post-training paradigm designed for rapid and stable adaptation. The pipeline consist in three stages: (1) Exploratory Learning on curated SNS corpora to establish initial alignment and identify systematic weaknesses; (2) Targeted Fine-Tuning that selectively applies SFT to the diagnosed gaps while mixing a small fraction of general data to mitigate forgetting; and (3) Refinement Learning that re-applies RL with SNS-centric signals to consolidate improvements and harmonize trade-offs across tasks. Across various tasks spanning three categories, our 4B scale model delivers an average improvements about 2.41 over the 7B sub-optimal baseline. Additionally, RedOne 2.0 achieves average performance lift about 8.74 from the base model with less than half the data required by SFT-centric method RedOne, evidencing superior data efficiency and stability at compact scales. Overall, RedOne 2.0 establishes a competitive, cost-effective baseline for domain-specific LLMs in SNS scenario, advancing capability without sacrificing robustness. 

---
# LPFQA: A Long-Tail Professional Forum-based Benchmark for LLM Evaluation 

**Authors**: Liya Zhu, Peizhuang Cong, Aowei Ji, Wenya Wu, Jiani Hou, Chunjie Wu, Xiang Gao, Jingkai Liu, Zhou Huan, Xuelei Sun, Yang Yang, Jianpeng Jiao, Liang Hu, Xinjie Chen, Jiashuo Liu, Jingzhe Ding, Tong Yang, Zaiyuan Wang, Ge Zhang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06346)  

**Abstract**: Large Language Models (LLMs) have made rapid progress in reasoning, question answering, and professional applications; however, their true capabilities remain difficult to evaluate using existing benchmarks. Current datasets often focus on simplified tasks or artificial scenarios, overlooking long-tail knowledge and the complexities of real-world applications. To bridge this gap, we propose LPFQA, a long-tail knowledge-based benchmark derived from authentic professional forums across 20 academic and industrial fields, covering 502 tasks grounded in practical expertise. LPFQA introduces four key innovations: fine-grained evaluation dimensions that target knowledge depth, reasoning, terminology comprehension, and contextual analysis; a hierarchical difficulty structure that ensures semantic clarity and unique answers; authentic professional scenario modeling with realistic user personas; and interdisciplinary knowledge integration across diverse domains. We evaluated 12 mainstream LLMs on LPFQA and observed significant performance disparities, especially in specialized reasoning tasks. LPFQA provides a robust, authentic, and discriminative benchmark for advancing LLM evaluation and guiding future model development. 

---
# Secu-Table: a Comprehensive security table dataset for evaluating semantic table interpretation systems 

**Authors**: Azanzi Jiomekong, Jean Bikim, Patricia Negoue, Joyce Chin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06301)  

**Abstract**: Evaluating semantic tables interpretation (STI) systems, (particularly, those based on Large Language Models- LLMs) especially in domain-specific contexts such as the security domain, depends heavily on the dataset. However, in the security domain, tabular datasets for state-of-the-art are not publicly available. In this paper, we introduce Secu-Table dataset, composed of more than 1500 tables with more than 15k entities constructed using security data extracted from Common Vulnerabilities and Exposures (CVE) and Common Weakness Enumeration (CWE) data sources and annotated using Wikidata and the SEmantic Processing of Security Event Streams CyberSecurity Knowledge Graph (SEPSES CSKG). Along with the dataset, all the code is publicly released. This dataset is made available to the research community in the context of the SemTab challenge on Tabular to Knowledge Graph Matching. This challenge aims to evaluate the performance of several STI based on open source LLMs. Preliminary evaluation, serving as baseline, was conducted using Falcon3-7b-instruct and Mistral-7B-Instruct, two open source LLMs and GPT-4o mini one closed source LLM. 

---
# GAIA: A General Agency Interaction Architecture for LLM-Human B2B Negotiation & Screening 

**Authors**: Siming Zhao, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06262)  

**Abstract**: Organizations are increasingly exploring delegation of screening and negotiation tasks to AI systems, yet deployment in high-stakes B2B settings is constrained by governance: preventing unauthorized commitments, ensuring sufficient information before bargaining, and maintaining effective human oversight and auditability. Prior work on large language model negotiation largely emphasizes autonomous bargaining between agents and omits practical needs such as staged information gathering, explicit authorization boundaries, and systematic feedback integration. We propose GAIA, a governance-first framework for LLM-human agency in B2B negotiation and screening. GAIA defines three essential roles - Principal (human), Delegate (LLM agent), and Counterparty - with an optional Critic to enhance performance, and organizes interactions through three mechanisms: information-gated progression that separates screening from negotiation; dual feedback integration that combines AI critique with lightweight human corrections; and authorization boundaries with explicit escalation paths. Our contributions are fourfold: (1) a formal governance framework with three coordinated mechanisms and four safety invariants for delegation with bounded authorization; (2) information-gated progression via task-completeness tracking (TCI) and explicit state transitions that separate screening from commitment; (3) dual feedback integration that blends Critic suggestions with human oversight through parallel learning channels; and (4) a hybrid validation blueprint that combines automated protocol metrics with human judgment of outcomes and safety. By bridging theory and practice, GAIA offers a reproducible specification for safe, efficient, and accountable AI delegation that can be instantiated across procurement, real estate, and staffing workflows. 

---
# Synthetic Data-Driven Prompt Tuning for Financial QA over Tables and Documents 

**Authors**: Yaoning Yu, Kaimin Chang, Ye Yu, Kai Wei, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06292)  

**Abstract**: Financial documents like earning reports or balance sheets often involve long tables and multi-page reports. Large language models have become a new tool to help numerical reasoning and understanding these documents. However, prompt quality can have a major effect on how well LLMs perform these financial reasoning tasks. Most current methods tune prompts on fixed datasets of financial text or tabular data, which limits their ability to adapt to new question types or document structures, or they involve costly and manually labeled/curated dataset to help build the prompts. We introduce a self-improving prompt framework driven by data-augmented optimization. In this closed-loop process, we generate synthetic financial tables and document excerpts, verify their correctness and robustness, and then update the prompt based on the results. Specifically, our framework combines a synthetic data generator with verifiers and a prompt optimizer, where the generator produces new examples that exposes weaknesses in the current prompt, the verifiers check the validity and robustness of the produced examples, and the optimizer incrementally refines the prompt in response. By iterating these steps in a feedback cycle, our method steadily improves prompt accuracy on financial reasoning tasks without needing external labels. Evaluation on DocMath-Eval benchmark demonstrates that our system achieves higher performance in both accuracy and robustness than standard prompt methods, underscoring the value of incorporating synthetic data generation into prompt learning for financial applications. 

---
# What Makes Reasoning Invalid: Echo Reflection Mitigation for Large Language Models 

**Authors**: Chen He, Xun Jiang, Lei Wang, Hao Yang, Chong Peng, Peng Yan, Fumin Shen, Xing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06380)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of reasoning tasks. Recent methods have further improved LLM performance in complex mathematical reasoning. However, when extending these methods beyond the domain of mathematical reasoning to tasks involving complex domain-specific knowledge, we observe a consistent failure of LLMs to generate novel insights during the reflection stage. Instead of conducting genuine cognitive refinement, the model tends to mechanically reiterate earlier reasoning steps without introducing new information or perspectives, a phenomenon referred to as "Echo Reflection". We attribute this behavior to two key defects: (1) Uncontrollable information flow during response generation, which allows premature intermediate thoughts to propagate unchecked and distort final decisions; (2) Insufficient exploration of internal knowledge during reflection, leading to repeating earlier findings rather than generating new cognitive insights. Building on these findings, we proposed a novel reinforcement learning method termed Adaptive Entropy Policy Optimization (AEPO). Specifically, the AEPO framework consists of two major components: (1) Reflection-aware Information Filtration, which quantifies the cognitive information flow and prevents the final answer from being affected by earlier bad cognitive information; (2) Adaptive-Entropy Optimization, which dynamically balances exploration and exploitation across different reasoning stages, promoting both reflective diversity and answer correctness. Extensive experiments demonstrate that AEPO consistently achieves state-of-the-art performance over mainstream reinforcement learning baselines across diverse benchmarks. 

---
# Chasing Consistency: Quantifying and Optimizing Human-Model Alignment in Chain-of-Thought Reasoning 

**Authors**: Boxuan Wang, Zhuoyun Li, Xinmiao Huang, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06168)  

**Abstract**: This paper presents a framework for evaluating and optimizing reasoning consistency in Large Language Models (LLMs) via a new metric, the Alignment Score, which quantifies the semantic alignment between model-generated reasoning chains and human-written reference chains in Chain-of-Thought (CoT) reasoning. Empirically, we find that 2-hop reasoning chains achieve the highest Alignment Score. To explain this phenomenon, we define four key error types: logical disconnection, thematic shift, redundant reasoning, and causal reversal, and show how each contributes to the degradation of the Alignment Score. Building on this analysis, we further propose Semantic Consistency Optimization Sampling (SCOS), a method that samples and favors chains with minimal alignment errors, significantly improving Alignment Scores by an average of 29.84% with longer reasoning chains, such as in 3-hop tasks. 

---
# Dataforge: A Data Agent Platform for Autonomous Data Engineering 

**Authors**: Xinyuan Wang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06185)  

**Abstract**: The growing demand for AI applications in fields such as materials discovery, molecular modeling, and climate science has made data preparation an important but labor-intensive step. Raw data from diverse sources must be cleaned, normalized, and transformed to become AI-ready, while effective feature transformation and selection are essential for efficient training and inference. To address the challenges of scalability and expertise dependence, we present Data Agent, a fully autonomous system specialized for tabular data. Leveraging large language model (LLM) reasoning and grounded validation, Data Agent automatically performs data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops. It embodies three core principles: automatic, safe, and non-expert friendly, which ensure end-to-end reliability without human supervision. This demo showcases the first practical realization of an autonomous Data Agent, illustrating how raw data can be transformed "From Data to Better Data." 

---
# Evaluating Implicit Biases in LLM Reasoning through Logic Grid Puzzles 

**Authors**: Fatima Jahara, Mark Dredze, Sharon Levy  

**Link**: [PDF](https://arxiv.org/pdf/2511.06160)  

**Abstract**: While recent safety guardrails effectively suppress overtly biased outputs, subtler forms of social bias emerge during complex logical reasoning tasks that evade current evaluation benchmarks. To fill this gap, we introduce a new evaluation framework, PRIME (Puzzle Reasoning for Implicit Biases in Model Evaluation), that uses logic grid puzzles to systematically probe the influence of social stereotypes on logical reasoning and decision making in LLMs. Our use of logic puzzles enables automatic generation and verification, as well as variability in complexity and biased settings. PRIME includes stereotypical, anti-stereotypical, and neutral puzzle variants generated from a shared puzzle structure, allowing for controlled and fine-grained comparisons. We evaluate multiple model families across puzzle sizes and test the effectiveness of prompt-based mitigation strategies. Focusing our experiments on gender stereotypes, our findings highlight that models consistently reason more accurately when solutions align with stereotypical associations. This demonstrates the significance of PRIME for diagnosing and quantifying social biases perpetuated in the deductive reasoning of LLMs, where fairness is critical. 

---
# Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads 

**Authors**: Jingwei Ni, Ekaterina Fadeeva, Tianyi Wu, Mubashara Akhtar, Jiaheng Zhang, Elliott Ash, Markus Leippold, Timothy Baldwin, See-Kiong Ng, Artem Shelmanov, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2511.06209)  

**Abstract**: Solving complex tasks usually requires LLMs to generate long multi-step reasoning chains. Previous work has shown that verifying the correctness of individual reasoning steps can further improve the performance and efficiency of LLMs on such tasks and enhance solution interpretability. However, existing verification approaches, such as Process Reward Models (PRMs), are either computationally expensive, limited to specific domains, or require large-scale human or model-generated annotations. Thus, we propose a lightweight alternative for step-level reasoning verification based on data-driven uncertainty scores. We train transformer-based uncertainty quantification heads (UHeads) that use the internal states of a frozen LLM to estimate the uncertainty of its reasoning steps during generation. The approach is fully automatic: target labels are generated either by another larger LLM (e.g., DeepSeek R1) or in a self-supervised manner by the original model itself. UHeads are both effective and lightweight, containing less than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, they match or even surpass the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their uncertainty and can serve as reliable signals for reasoning verification, offering a promising direction toward scalable and generalizable introspective LLMs. 

---
# Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B 

**Authors**: Sen Xu, Yi Zhou, Wei Wang, Jixin Min, Zhibin Yin, Yingwei Dai, Shixi Liu, Lianyu Pang, Yirong Chen, Junlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06221)  

**Abstract**: Challenging the prevailing consensus that small models inherently lack robust reasoning, this report introduces VibeThinker-1.5B, a 1.5B-parameter dense model developed via our Spectrum-to-Signal Principle (SSP). This challenges the prevailing approach of scaling model parameters to enhance capabilities, as seen in models like DeepSeek R1 (671B) and Kimi k2 (>1T). The SSP framework first employs a Two-Stage Diversity-Exploring Distillation (SFT) to generate a broad spectrum of solutions, followed by MaxEnt-Guided Policy Optimization (RL) to amplify the correct signal. With a total training cost of only $7,800, VibeThinker-1.5B demonstrates superior reasoning capabilities compared to closed-source models like Magistral Medium and Claude Opus 4, and performs on par with open-source models like GPT OSS-20B Medium. Remarkably, it surpasses the 400x larger DeepSeek R1 on three math benchmarks: AIME24 (80.3 vs. 79.8), AIME25 (74.4 vs. 70.0), and HMMT25 (50.4 vs. 41.7). This is a substantial improvement over its base model (6.7, 4.3, and 0.6, respectively). On LiveCodeBench V6, it scores 51.1, outperforming Magistral Medium's 50.3 and its base model's 0.0. These findings demonstrate that small models can achieve reasoning capabilities comparable to large models, drastically reducing training and inference costs and thereby democratizing advanced AI research. 

---
# CSP4SDG: Constraint and Information-Theory Based Role Identification in Social Deduction Games with LLM-Enhanced Inference 

**Authors**: Kaijie Xu, Fandi Meng, Clark Verbrugge, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2511.06175)  

**Abstract**: In Social Deduction Games (SDGs) such as Avalon, Mafia, and Werewolf, players conceal their identities and deliberately mislead others, making hidden-role inference a central and demanding task. Accurate role identification, which forms the basis of an agent's belief state, is therefore the keystone for both human and AI performance. We introduce CSP4SDG, a probabilistic, constraint-satisfaction framework that analyses gameplay objectively. Game events and dialogue are mapped to four linguistically-agnostic constraint classes-evidence, phenomena, assertions, and hypotheses. Hard constraints prune impossible role assignments, while weighted soft constraints score the remainder; information-gain weighting links each hypothesis to its expected value under entropy reduction, and a simple closed-form scoring rule guarantees that truthful assertions converge to classical hard logic with minimum error. The resulting posterior over roles is fully interpretable and updates in real time. Experiments on three public datasets show that CSP4SDG (i) outperforms LLM-based baselines in every inference scenario, and (ii) boosts LLMs when supplied as an auxiliary "reasoning tool." Our study validates that principled probabilistic reasoning with information theory is a scalable alternative-or complement-to heavy-weight neural models for SDGs. 

---
# An Empirical Study of Reasoning Steps in Thinking Code LLMs 

**Authors**: Haoran Xue, Gias Uddin, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05874)  

**Abstract**: Thinking Large Language Models (LLMs) generate explicit intermediate reasoning traces before final answers, potentially improving transparency, interpretability, and solution accuracy for code generation. However, the quality of these reasoning chains remains underexplored. We present a comprehensive empirical study examining the reasoning process and quality of thinking LLMs for code generation. We evaluate six state-of-the-art reasoning LLMs (DeepSeek-R1, OpenAI-o3-mini, Claude-3.7-Sonnet-Thinking, Gemini-2.0-Flash-Thinking, Gemini-2.5-Flash, and Qwen-QwQ) across 100 code generation tasks of varying difficulty from BigCodeBench. We quantify reasoning-chain structure through step counts and verbosity, conduct controlled step-budget adjustments, and perform a 21-participant human evaluation across three dimensions: efficiency, logical correctness, and completeness. Our step-count interventions reveal that targeted step increases can improve resolution rates for certain models/tasks, while modest reductions often preserve success on standard tasks, rarely on hard ones. Through systematic analysis, we develop a reasoning-problematic taxonomy, identifying completeness as the dominant failure mode. Task complexity significantly impacts reasoning quality; hard problems are substantially more prone to incompleteness than standard tasks. Our stability analysis demonstrates that thinking LLMs maintain consistent logical structures across computational effort levels and can self-correct previous errors. This study provides new insights into the strengths and limitations of current thinking LLMs in software engineering. 

---
# Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs 

**Authors**: Felipe Valencia-Clavijo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05766)  

**Abstract**: Large language models (LLMs) are increasingly examined as both behavioral subjects and decision systems, yet it remains unclear whether observed cognitive biases reflect surface imitation or deeper probability shifts. Anchoring bias, a classic human judgment bias, offers a critical test case. While prior work shows LLMs exhibit anchoring, most evidence relies on surface-level outputs, leaving internal mechanisms and attributional contributions unexplored. This paper advances the study of anchoring in LLMs through three contributions: (1) a log-probability-based behavioral analysis showing that anchors shift entire output distributions, with controls for training-data contamination; (2) exact Shapley-value attribution over structured prompt fields to quantify anchor influence on model log-probabilities; and (3) a unified Anchoring Bias Sensitivity Score integrating behavioral and attributional evidence across six open-source models. Results reveal robust anchoring effects in Gemma-2B, Phi-2, and Llama-2-7B, with attribution signaling that the anchors influence reweighting. Smaller models such as GPT-2, Falcon-RW-1B, and GPT-Neo-125M show variability, suggesting scale may modulate sensitivity. Attributional effects, however, vary across prompt designs, underscoring fragility in treating LLMs as human substitutes. The findings demonstrate that anchoring bias in LLMs is robust, measurable, and interpretable, while highlighting risks in applied domains. More broadly, the framework bridges behavioral science, LLM safety, and interpretability, offering a reproducible path for evaluating other cognitive biases in LLMs. 

---
# CoT-X: An Adaptive Framework for Cross-Model Chain-of-Thought Transfer and Optimization 

**Authors**: Ziqian Bi, Kaijie Chen, Tianyang Wang, Junfeng Hao, Xinyuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.05747)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances the problem-solving ability of large language models (LLMs) but leads to substantial inference overhead, limiting deployment in resource-constrained settings. This paper investigates efficient CoT transfer across models of different scales and architectures through an adaptive reasoning summarization framework. The proposed method compresses reasoning traces via semantic segmentation with importance scoring, budget-aware dynamic compression, and coherence reconstruction, preserving critical reasoning steps while significantly reducing token usage. Experiments on 7{,}501 medical examination questions across 10 specialties show up to 40% higher accuracy than truncation under the same token budgets. Evaluations on 64 model pairs from eight LLMs (1.5B-32B parameters, including DeepSeek-R1 and Qwen3) confirm strong cross-model transferability. Furthermore, a Gaussian Process-based Bayesian optimization module reduces evaluation cost by 84% and reveals a power-law relationship between model size and cross-domain robustness. These results demonstrate that reasoning summarization provides a practical path toward efficient CoT transfer, enabling advanced reasoning under tight computational constraints. Code will be released upon publication. 

---
# Self-Abstraction from Grounded Experience for Plan-Guided Policy Refinement 

**Authors**: Hiroaki Hayashi, Bo Pang, Wenting Zhao, Ye Liu, Akash Gokul, Srijan Bansal, Caiming Xiong, Semih Yavuz, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.05931)  

**Abstract**: Large language model (LLM) based agents are increasingly used to tackle software engineering tasks that require multi-step reasoning and code modification, demonstrating promising yet limited performance. However, most existing LLM agents typically operate within static execution frameworks, lacking a principled mechanism to learn and self-improve from their own experience and past rollouts. As a result, their performance remains bounded by the initial framework design and the underlying LLM's capabilities. We propose Self-Abstraction from Grounded Experience (SAGE), a framework that enables agents to learn from their own task executions and refine their behavior through self-abstraction. After an initial rollout, the agent induces a concise plan abstraction from its grounded experience, distilling key steps, dependencies, and constraints. This learned abstraction is then fed back as contextual guidance, refining the agent's policy and supporting more structured, informed subsequent executions. Empirically, SAGE delivers consistent performance gains across diverse LLM backbones and agent architectures. Notably, it yields a 7.2% relative performance improvement over the strong Mini-SWE-Agent baseline when paired with the GPT-5 (high) backbone. SAGE further achieves strong overall performance on SWE-Bench Verified benchmark, reaching 73.2% and 74% Pass@1 resolve rates with the Mini-SWE-Agent and OpenHands CodeAct agent framework, respectively. 

---
# DiagnoLLM: A Hybrid Bayesian Neural Language Framework for Interpretable Disease Diagnosis 

**Authors**: Bowen Xu, Xinyue Zeng, Jiazhen Hu, Tuo Wang, Adithya Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2511.05810)  

**Abstract**: Building trustworthy clinical AI systems requires not only accurate predictions but also transparent, biologically grounded explanations. We present \texttt{DiagnoLLM}, a hybrid framework that integrates Bayesian deconvolution, eQTL-guided deep learning, and LLM-based narrative generation for interpretable disease diagnosis. DiagnoLLM begins with GP-unmix, a Gaussian Process-based hierarchical model that infers cell-type-specific gene expression profiles from bulk and single-cell RNA-seq data while modeling biological uncertainty. These features, combined with regulatory priors from eQTL analysis, power a neural classifier that achieves high predictive performance in Alzheimer's Disease (AD) detection (88.0\% accuracy). To support human understanding and trust, we introduce an LLM-based reasoning module that translates model outputs into audience-specific diagnostic reports, grounded in clinical features, attribution signals, and domain knowledge. Human evaluations confirm that these reports are accurate, actionable, and appropriately tailored for both physicians and patients. Our findings show that LLMs, when deployed as post-hoc reasoners rather than end-to-end predictors, can serve as effective communicators within hybrid diagnostic pipelines. 

---
# Can a Small Model Learn to Look Before It Leaps? Dynamic Learning and Proactive Correction for Hallucination Detection 

**Authors**: Zepeng Bao, Shen Zhou, Qiankun Pi, Jianhao Chen, Mayi Xu, Ming Zhong, Yuanyuan Zhu, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.05854)  

**Abstract**: Hallucination in large language models (LLMs) remains a critical barrier to their safe deployment. Existing tool-augmented hallucination detection methods require pre-defined fixed verification strategies, which are crucial to the quality and effectiveness of tool calls. Some methods directly employ powerful closed-source LLMs such as GPT-4 as detectors, which are effective but too costly. To mitigate the cost issue, some methods adopt the teacher-student architecture and finetune open-source small models as detectors via agent tuning. However, these methods are limited by fixed strategies. When faced with a dynamically changing execution environment, they may lack adaptability and inappropriately call tools, ultimately leading to detection failure. To address the problem of insufficient strategy adaptability, we propose the innovative ``Learning to Evaluate and Adaptively Plan''(LEAP) framework, which endows an efficient student model with the dynamic learning and proactive correction capabilities of the teacher model. Specifically, our method formulates the hallucination detection problem as a dynamic strategy learning problem. We first employ a teacher model to generate trajectories within the dynamic learning loop and dynamically adjust the strategy based on execution failures. We then distill this dynamic planning capability into an efficient student model via agent tuning. Finally, during strategy execution, the student model adopts a proactive correction mechanism, enabling it to propose, review, and optimize its own verification strategies before execution. We demonstrate through experiments on three challenging benchmarks that our LEAP-tuned model outperforms existing state-of-the-art methods. 

---
# Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction 

**Authors**: Hyeryun Park, Byung Mo Gu, Jun Hee Lee, Byeong Hyeon Choi, Sekeun Kim, Hyun Koo Kim, Kyungsang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07392)  

**Abstract**: In da Vinci robotic surgery, surgeons' hands and eyes are fully engaged in the procedure, making it difficult to access and manipulate multimodal patient data without interruption. We propose a voice-directed Surgical Agent Orchestrator Platform (SAOP) built on a hierarchical multi-agent framework, consisting of an orchestration agent and three task-specific agents driven by Large Language Models (LLMs). These LLM-based agents autonomously plan, refine, validate, and reason to map voice commands into specific tasks such as retrieving clinical information, manipulating CT scans, or navigating 3D anatomical models on the surgical video. We also introduce a Multi-level Orchestration Evaluation Metric (MOEM) to comprehensively assess the performance and robustness from command-level and category-level perspectives. The SAOP achieves high accuracy and success rates across 240 voice commands, while LLM-based agents improve robustness against speech recognition errors and diverse or ambiguous free-form commands, demonstrating strong potential to support minimally invasive da Vinci robotic surgery. 

---
# Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection 

**Authors**: Vaibhav Mavi, Shubh Jaroria, Weiqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.07364)  

**Abstract**: Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection. 

---
# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards 

**Authors**: Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2511.07403)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning. 

---
# ScRPO: From Errors to Insights 

**Authors**: Lianrui Li, Dakuan Lu, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06065)  

**Abstract**: We propose Self-correction Relative Policy Optimization (ScRPO), a novel reinforcement learning framework designed to enhance large language models on challenging mathemati- cal problems by leveraging self-reflection and error correction. Our approach consists of two stages: (1) Trial-and-error learning stage: training the model with GRPO and collect- ing incorrect answers along with their cor- responding questions in an error pool; (2) Self-correction learning stage: guiding the model to reflect on why its previous an- swers were wrong. Extensive experiments across multiple math reasoning benchmarks, including AIME, AMC, Olympiad, MATH- 500, GSM8k, using Deepseek-Distill-Qwen- 1.5B and Deepseek-Distill-Qwen-7B. The ex- perimental results demonstrate that ScRPO consistently outperforms several post-training methods. These findings highlight ScRPO as a promising paradigm for enabling language models to self-improve on difficult tasks with limited external feedback, paving the way to- ward more reliable and capable AI systems. 

---
# Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence 

**Authors**: Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2511.07384)  

**Abstract**: Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute. In this work, we study how to convert existing pretrained non-recurrent language models into depth-recurrent models. We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost. In our experiments, on mathematics, we observe that converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model. 

---
# Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization 

**Authors**: Yu Huang, Zixin Wen, Aarti Singh, Yuejie Chi, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07378)  

**Abstract**: The ability to reason lies at the core of artificial intelligence (AI), and challenging problems usually call for deeper and longer reasoning to tackle. A crucial question about AI reasoning is whether models can extrapolate learned reasoning patterns to solve harder tasks with longer chain-of-thought (CoT). In this work, we present a theoretical analysis of transformers learning on synthetic state-tracking tasks with gradient descent. We mathematically prove how the algebraic structure of state-tracking problems governs the degree of extrapolation of the learned CoT. Specifically, our theory characterizes the length generalization of transformers through the mechanism of attention concentration, linking the retrieval robustness of the attention layer to the state-tracking task structure of long-context reasoning. Moreover, for transformers with limited reasoning length, we prove that a recursive self-training scheme can progressively extend the range of solvable problem lengths. To our knowledge, we provide the first optimization guarantee that constant-depth transformers provably learn $\mathsf{NC}^1$-complete problems with CoT, significantly going beyond prior art confined in $\mathsf{TC}^0$, unless the widely held conjecture $\mathsf{TC}^0 \neq \mathsf{NC}^1$ fails. Finally, we present a broad set of experiments supporting our theoretical results, confirming the length generalization behaviors and the mechanism of attention concentration. 

---
# FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation 

**Authors**: Song Jin, Shuqi Li, Shukun Zhang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07322)  

**Abstract**: While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available. 

---
# When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs 

**Authors**: Shaowen Wang, Yiqi Dong, Ruinian Chang, Tansheng Zhu, Yuebo Sun, Kaifeng Lyu, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.07318)  

**Abstract**: Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations. 

---
# Discourse Graph Guided Document Translation with Large Language Models 

**Authors**: Viet-Thanh Pham, Minghan Wang, Hao-Han Liao, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07230)  

**Abstract**: Adapting large language models to full document translation remains challenging due to the difficulty of capturing long-range dependencies and preserving discourse coherence throughout extended texts. While recent agentic machine translation systems mitigate context window constraints through multi-agent orchestration and persistent memory, they require substantial computational resources and are sensitive to memory retrieval strategies. We introduce TransGraph, a discourse-guided framework that explicitly models inter-chunk relationships through structured discourse graphs and selectively conditions each translation segment on relevant graph neighbourhoods rather than relying on sequential or exhaustive context. Across three document-level MT benchmarks spanning six languages and diverse domains, TransGraph consistently surpasses strong baselines in translation quality and terminology consistency while incurring significantly lower token overhead. 

---
# AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning 

**Authors**: Meiyun Wang, Charin Polpanumas  

**Link**: [PDF](https://arxiv.org/pdf/2511.07166)  

**Abstract**: We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks. 

---
# Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought 

**Authors**: Zhikang Chen, Sen Cui, Deheng Ye, Yu Zhang, Yatao Bian, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07124)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities through \emph{Chain-of-Thought} (CoT) prompting, which enables step-by-step intermediate reasoning. However, explicit CoT methods rely on discrete token-level reasoning processes that are prone to error propagation and limited by vocabulary expressiveness, often resulting in rigid and inconsistent reasoning trajectories. Recent research has explored implicit or continuous reasoning in latent spaces, allowing models to perform internal reasoning before generating explicit output. Although such approaches alleviate some limitations of discrete CoT, they generally lack explicit mechanisms to enforce consistency among reasoning steps, leading to divergent reasoning paths and unstable outcomes. To address this issue, we propose EBM-CoT, an Energy-Based Chain-of-Thought Calibration framework that refines latent thought representations through an energy-based model (EBM). Our method dynamically adjusts latent reasoning trajectories toward lower-energy, high-consistency regions in the embedding space, improving both reasoning accuracy and consistency without modifying the base language model. Extensive experiments across mathematical, commonsense, and symbolic reasoning benchmarks demonstrate that the proposed framework significantly enhances the consistency and efficiency of multi-step reasoning in LLMs. 

---
# Achieving Effective Virtual Reality Interactions via Acoustic Gesture Recognition based on Large Language Models 

**Authors**: Xijie Zhang, Fengliang He, Hong-Ning Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07085)  

**Abstract**: Natural and efficient interaction remains a critical challenge for virtual reality and augmented reality (VR/AR) systems. Vision-based gesture recognition suffers from high computational cost, sensitivity to lighting conditions, and privacy leakage concerns. Acoustic sensing provides an attractive alternative: by emitting inaudible high-frequency signals and capturing their reflections, channel impulse response (CIR) encodes how gestures perturb the acoustic field in a low-cost and user-transparent manner. However, existing CIR-based gesture recognition methods often rely on extensive training of models on large labeled datasets, making them unsuitable for few-shot VR scenarios. In this work, we propose the first framework that leverages large language models (LLMs) for CIR-based gesture recognition in VR/AR systems. Despite LLMs' strengths, it is non-trivial to achieve few-shot and zero-shot learning of CIR gestures due to their inconspicuous features. To tackle this challenge, we collect differential CIR rather than original CIR data. Moreover, we construct a real-world dataset collected from 10 participants performing 15 gestures across three categories (digits, letters, and shapes), with 10 repetitions each. We then conduct extensive experiments on this dataset using an LLM-adopted classifier. Results show that our LLM-based framework achieves accuracy comparable to classical machine learning baselines, while requiring no domain-specific retraining. 

---
# Benchmarking LLMs for Fine-Grained Code Review with Enriched Context in Practice 

**Authors**: Ruida Hu, Xinchen Wang, Xin-Cheng Wen, Zhao Zhang, Bo Jiang, Pengfei Gao, Chao Peng, Cuiyun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07017)  

**Abstract**: Code review is a cornerstone of software quality assurance, and recent advances in Large Language Models (LLMs) have shown promise in automating this process. However, existing benchmarks for LLM-based code review face three major limitations. (1) Lack of semantic context: most benchmarks provide only code diffs without textual information such as issue descriptions, which are crucial for understanding developer intent. (2) Data quality issues: without rigorous validation, many samples are noisy-e.g., reviews on outdated or irrelevant code-reducing evaluation reliability. (3) Coarse granularity: most benchmarks operate at the file or commit level, overlooking the fine-grained, line-level reasoning essential for precise review.
We introduce ContextCRBench, a high-quality, context-rich benchmark for fine-grained LLM evaluation in code review. Our construction pipeline comprises: (1) Raw Data Crawling, collecting 153.7K issues and pull requests from top-tier repositories; (2) Comprehensive Context Extraction, linking issue-PR pairs for textual context and extracting the full surrounding function or class for code context; and (3) Multi-stage Data Filtering, combining rule-based and LLM-based validation to remove outdated, malformed, or low-value samples, resulting in 67,910 context-enriched entries.
ContextCRBench supports three evaluation scenarios aligned with the review workflow: (1) hunk-level quality assessment, (2) line-level defect localization, and (3) line-level comment generation. Evaluating eight leading LLMs (four closed-source and four open-source) reveals that textual context yields greater performance gains than code context alone, while current LLMs remain far from human-level review ability. Deployed at ByteDance, ContextCRBench drives a self-evolving code review system, improving performance by 61.98% and demonstrating its robustness and industrial utility. 

---
# AgentSUMO: An Agentic Framework for Interactive Simulation Scenario Generation in SUMO via Large Language Models 

**Authors**: Minwoo Jeong, Jeeyun Chang, Yoonjin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.06804)  

**Abstract**: The growing complexity of urban mobility systems has made traffic simulation indispensable for evidence-based transportation planning and policy evaluation. However, despite the analytical capabilities of platforms such as the Simulation of Urban MObility (SUMO), their application remains largely confined to domain experts. Developing realistic simulation scenarios requires expertise in network construction, origin-destination modeling, and parameter configuration for policy experimentation, creating substantial barriers for non-expert users such as policymakers, urban planners, and city officials. Moreover, the requests expressed by these users are often incomplete and abstract-typically articulated as high-level objectives, which are not well aligned with the imperative, sequential workflows employed in existing language-model-based simulation frameworks. To address these challenges, this study proposes AgentSUMO, an agentic framework for interactive simulation scenario generation via large language models. AgentSUMO departs from imperative, command-driven execution by introducing an adaptive reasoning layer that interprets user intents, assesses task complexity, infers missing parameters, and formulates executable simulation plans. The framework is structured around two complementary components, the Interactive Planning Protocol, which governs reasoning and user interaction, and the Model Context Protocol, which manages standardized communication and orchestration among simulation tools. Through this design, AgentSUMO converts abstract policy objectives into executable simulation scenarios. Experiments on urban networks in Seoul and Manhattan demonstrate that the agentic workflow achieves substantial improvements in traffic flow metrics while maintaining accessibility for non-expert users, successfully bridging the gap between policy goals and executable simulation workflows. 

---
# Data Trajectory Alignment for LLM Domain Adaptation: A Two-Phase Synthesis Framework for Telecommunications Mathematics 

**Authors**: Zhicheng Zhou, Jing Li, Suming Qiu, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.06776)  

**Abstract**: General-purpose large language models (LLMs) are increasingly deployed in verticals such as telecommunications, where adaptation is hindered by scarce, low-information-density corpora and tight mobile/edge constraints. We propose Data Trajectory Alignment (DTA), a two-phase, model-agnostic data curation framework that treats solution processes - not only final answers - as first-class supervision. Phase I (Initializing) synthesizes diverse, high-coverage candidates using an ensemble of strong teachers. Phase II (DTA) rewrites teacher solutions to align intermediate steps and presentation style with the target student's inductive biases and then performs signal-aware exemplar selection via agreement checks and reflection-based judging. Instantiated on telecommunications mathematics (e.g., link budgets, SNR/AMC selection, and power-control feasibility), DTA yields state-of-the-art (SOTA) accuracy on TELEMATH without enabling explicit "thinking" modes: 72.45% pass@1, surpassing distilled-only training by +17.65 points and outperforming a strong baseline (Qwen3-32B with thinking enabled) by +2.94 points. Token-shift analyses indicate that DTA concentrates gains on logical-structural discourse markers rather than merely amplifying domain nouns, indicating improved reasoning scaffolding. Under edge-like inference settings, DTA improves efficiency by reducing reliance on multi-sample voting and disabling expensive reasoning heuristics, cutting energy per output token by ~42% versus Qwen3-32B (thinking mode enabled) and end-to-end latency by ~60% versus Qwen3-32B (thinking mode disabled). These results demonstrate that aligning how solutions are produced enables compact, high-yield supervision that is effective for both accuracy and efficiency, offering a practical recipe for domain adaptation in low-resource verticals beyond telecom. 

---
# Rank-1 LoRAs Encode Interpretable Reasoning Signals 

**Authors**: Jake Ward, Paul Riechers, Adam Shai  

**Link**: [PDF](https://arxiv.org/pdf/2511.06739)  

**Abstract**: Reasoning models leverage inference-time compute to significantly enhance the performance of language models on difficult logical tasks, and have become a dominating paradigm in frontier LLMs. Despite their wide adoption, the mechanisms underpinning the enhanced performance of these reasoning models are not well understood. In this work, we show that the majority of new capabilities in reasoning models can be elicited by small, single-rank changes to base model parameters, with many of these changes being interpretable. Specifically, we use a rank-1 LoRA to create a minimal parameter adapter for Qwen-2.5-32B-Instruct which recovers 73-90% of reasoning-benchmark performance compared to a full parameter finetune. We find that the activations of this LoRA are as interpretable as MLP neurons, and fire for reasoning-specific behaviors. Finally, we train a sparse autoencoder on the entire activation state of this LoRA and identify fine-grained and monosemantic features. Our findings highlight that reasoning performance can arise largely from minimal changes to base model parameters, and explore what these changes affect. More broadly, our work shows that parameter-efficient training methods can be used as a targeted lens for uncovering fundamental insights about language model behavior and dynamics. 

---
# S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning 

**Authors**: Jiangwen Dong, Zehui Lin, Wanyu Lin, Mingjin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06727)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance in complex reasoning problems. Their effectiveness highly depends on the specific nature of the task, especially the required domain knowledge. Existing approaches, such as mixture-of-experts, typically operate at the task level; they are too coarse to effectively solve the heterogeneous problems involving multiple subjects. This work proposes a novel framework that performs fine-grained analysis at subject level equipped with a designated multi-agent collaboration strategy for addressing heterogeneous problem reasoning. Specifically, given an input query, we first employ a Graph Neural Network to identify the relevant subjects and infer their interdependencies to generate an \textit{Subject-based Directed Acyclic Graph} (S-DAG), where nodes represent subjects and edges encode information flow. Then we profile the LLM models by assigning each model a subject-specific expertise score, and select the top-performing one for matching corresponding subject of the S-DAG. Such subject-model matching enables graph-structured multi-agent collaboration where information flows from the starting model to the ending model over S-DAG. We curate and release multi-subject subsets of standard benchmarks (MMLU-Pro, GPQA, MedMCQA) to better reflect complex, real-world reasoning tasks. Extensive experiments show that our approach significantly outperforms existing task-level model selection and multi-agent collaboration baselines in accuracy and efficiency. These results highlight the effectiveness of subject-aware reasoning and structured collaboration in addressing complex and multi-subject problems. 

---
# Place Matters: Comparing LLM Hallucination Rates for Place-Based Legal Queries 

**Authors**: Damian Curran, Vanessa Sporne, Lea Frermann, Jeannie Paterson  

**Link**: [PDF](https://arxiv.org/pdf/2511.06700)  

**Abstract**: How do we make a meaningful comparison of a large language model's knowledge of the law in one place compared to another? Quantifying these differences is critical to understanding if the quality of the legal information obtained by users of LLM-based chatbots varies depending on their location. However, obtaining meaningful comparative metrics is challenging because legal institutions in different places are not themselves easily comparable. In this work we propose a methodology to obtain place-to-place metrics based on the comparative law concept of functionalism. We construct a dataset of factual scenarios drawn from Reddit posts by users seeking legal advice for family, housing, employment, crime and traffic issues. We use these to elicit a summary of a law from the LLM relevant to each scenario in Los Angeles, London and Sydney. These summaries, typically of a legislative provision, are manually evaluated for hallucinations. We show that the rate of hallucination of legal information by leading closed-source LLMs is significantly associated with place. This suggests that the quality of legal solutions provided by these models is not evenly distributed across geography. Additionally, we show a strong negative correlation between hallucination rate and the frequency of the majority response when the LLM is sampled multiple times, suggesting a measure of uncertainty of model predictions of legal facts. 

---
# CoFineLLM: Conformal Finetuning of LLMs for Language-Instructed Robot Planning 

**Authors**: Jun Wang, Yevgeniy Vorobeychik, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2511.06575)  

**Abstract**: Large Language Models (LLMs) have recently emerged as planners for language-instructed agents, generating sequences of actions to accomplish natural language tasks. However, their reliability remains a challenge, especially in long-horizon tasks, since they often produce overconfident yet wrong outputs. Conformal Prediction (CP) has been leveraged to address this issue by wrapping LLM outputs into prediction sets that contain the correct action with a user-defined confidence. When the prediction set is a singleton, the planner executes that action; otherwise, it requests help from a user. This has led to LLM-based planners that can ensure plan correctness with a user-defined probability. However, as LLMs are trained in an uncertainty-agnostic manner, without awareness of prediction sets, they tend to produce unnecessarily large sets, particularly at higher confidence levels, resulting in frequent human interventions limiting autonomous deployment. To address this, we introduce CoFineLLM (Conformal Finetuning for LLMs), the first CP-aware finetuning framework for LLM-based planners that explicitly reduces prediction-set size and, in turn, the need for user interventions. We evaluate our approach on multiple language-instructed robot planning problems and show consistent improvements over uncertainty-aware and uncertainty-agnostic finetuning baselines in terms of prediction-set size, and help rates. Finally, we demonstrate robustness of our method to out-of-distribution scenarios in hardware experiments. 

---
# Rep2Text: Decoding Full Text from a Single LLM Token Representation 

**Authors**: Haiyan Zhao, Zirui He, Fan Yang, Ali Payani, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.06571)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress across diverse tasks, yet their internal mechanisms remain largely opaque. In this work, we address a fundamental question: to what extent can the original input text be recovered from a single last-token representation within an LLM? We propose Rep2Text, a novel framework for decoding full text from last-token representations. Rep2Text employs a trainable adapter that projects a target model's internal representations into the embedding space of a decoding language model, which then autoregressively reconstructs the input text. Experiments on various model combinations (Llama-3.1-8B, Gemma-7B, Mistral-7B-v0.1, Llama-3.2-3B) demonstrate that, on average, over half of the information in 16-token sequences can be recovered from this compressed representation while maintaining strong semantic integrity and coherence. Furthermore, our analysis reveals an information bottleneck effect: longer sequences exhibit decreased token-level recovery while preserving strong semantic integrity. Besides, our framework also demonstrates robust generalization to out-of-distribution medical data. 

---
# On the Analogy between Human Brain and LLMs: Spotting Key Neurons in Grammar Perception 

**Authors**: Sanaz Saki Norouzi, Mohammad Masjedi, Pascal Hitzler  

**Link**: [PDF](https://arxiv.org/pdf/2511.06519)  

**Abstract**: Artificial Neural Networks, the building blocks of AI, were inspired by the human brain's network of neurons. Over the years, these networks have evolved to replicate the complex capabilities of the brain, allowing them to handle tasks such as image and language processing. In the realm of Large Language Models, there has been a keen interest in making the language learning process more akin to that of humans. While neuroscientific research has shown that different grammatical categories are processed by different neurons in the brain, we show that LLMs operate in a similar way. Utilizing Llama 3, we identify the most important neurons associated with the prediction of words belonging to different part-of-speech tags. Using the achieved knowledge, we train a classifier on a dataset, which shows that the activation patterns of these key neurons can reliably predict part-of-speech tags on fresh data. The results suggest the presence of a subspace in LLMs focused on capturing part-of-speech tag concepts, resembling patterns observed in lesion studies of the brain in neuroscience. 

---
# LLM For Loop Invariant Generation and Fixing: How Far Are We? 

**Authors**: Mostafijur Rahman Akhond, Saikat Chakraborty, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06552)  

**Abstract**: A loop invariant is a property of a loop that remains true before and after each execution of the loop. The identification of loop invariants is a critical step to support automated program safety assessment. Recent advancements in Large Language Models (LLMs) have demonstrated potential in diverse software engineering (SE) and formal verification tasks. However, we are not aware of the performance of LLMs to infer loop invariants. We report an empirical study of both open-source and closed-source LLMs of varying sizes to assess their proficiency in inferring inductive loop invariants for programs and in fixing incorrect invariants. Our findings reveal that while LLMs exhibit some utility in inferring and repairing loop invariants, their performance is substantially enhanced when supplemented with auxiliary information such as domain knowledge and illustrative examples. LLMs achieve a maximum success rate of 78\% in generating, but are limited to 16\% in repairing the invariant. 

---
# A Multi-Agent System for Semantic Mapping of Relational Data to Knowledge Graphs 

**Authors**: Milena Trajanoska, Riste Stojanov, Dimitar Trajanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.06455)  

**Abstract**: Enterprises often maintain multiple databases for storing critical business data in siloed systems, resulting in inefficiencies and challenges with data interoperability. A key to overcoming these challenges lies in integrating disparate data sources, enabling businesses to unlock the full potential of their data. Our work presents a novel approach for integrating multiple databases using knowledge graphs, focusing on the application of large language models as semantic agents for mapping and connecting structured data across systems by leveraging existing vocabularies. The proposed methodology introduces a semantic layer above tables in relational databases, utilizing a system comprising multiple LLM agents that map tables and columns to this http URL terms. Our approach achieves a mapping accuracy of over 90% in multiple domains. 

---
# FLEX: Continuous Agent Evolution via Forward Learning from Experience 

**Authors**: Zhicheng Cai, Xinyuan Guo, Yu Pei, JiangTao Feng, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.06449)  

**Abstract**: Autonomous agents driven by Large Language Models (LLMs) have revolutionized reasoning and problem-solving but remain static after training, unable to grow with experience as intelligent beings do during deployment. We introduce Forward Learning with EXperience (FLEX), a gradient-free learning paradigm that enables LLM agents to continuously evolve through accumulated experience. Specifically, FLEX cultivates scalable and inheritable evolution by constructing a structured experience library through continual reflection on successes and failures during interaction with the environment. FLEX delivers substantial improvements on mathematical reasoning, chemical retrosynthesis, and protein fitness prediction (up to 23% on AIME25, 10% on USPTO50k, and 14% on ProteinGym). We further identify a clear scaling law of experiential growth and the phenomenon of experience inheritance across agents, marking a step toward scalable and inheritable continuous agent evolution. Project Page: this https URL. 

---
# When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms 

**Authors**: Qibing Ren, Zhijie Zheng, Jiaxuan Guo, Junchi Yan, Lizhuang Ma, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2511.06448)  

**Abstract**: In this work, we study the risks of collective financial fraud in large-scale multi-agent systems powered by large language model (LLM) agents. We investigate whether agents can collaborate in fraudulent behaviors, how such collaboration amplifies risks, and what factors influence fraud success. To support this research, we present MultiAgentFraudBench, a large-scale benchmark for simulating financial fraud scenarios based on realistic online interactions. The benchmark covers 28 typical online fraud scenarios, spanning the full fraud lifecycle across both public and private domains. We further analyze key factors affecting fraud success, including interaction depth, activity level, and fine-grained collaboration failure modes. Finally, we propose a series of mitigation strategies, including adding content-level warnings to fraudulent posts and dialogues, using LLMs as monitors to block potentially malicious agents, and fostering group resilience through information sharing at the societal level. Notably, we observe that malicious agents can adapt to environmental interventions. Our findings highlight the real-world risks of multi-agent financial fraud and suggest practical measures for mitigating them. Code is available at this https URL. 

---
# Route Experts by Sequence, not by Token 

**Authors**: Tiansheng Wen, Yifei Wang, Aosong Feng, Long Ma, Xinyang Liu, Yifan Wang, Lixuan Guo, Bo Chen, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2511.06494)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale large language models (LLMs) by activating only a subset of experts per token, but the standard TopK routing assigns the same fixed number of experts to all tokens, ignoring their varying complexity. Prior adaptive routing methods introduce additional modules and hyperparameters, often requiring costly retraining from scratch. We propose Sequence-level TopK (SeqTopK), a minimal modification that shifts the expert budget from the token level to the sequence level. By selecting the top $T \cdot K$ experts across all $T$ tokens, SeqTopK enables end-to-end learned dynamic allocation -- assigning more experts to difficult tokens and fewer to easy ones -- while preserving the same overall budget. SeqTopK requires only a few lines of code, adds less than 1% overhead, and remains fully compatible with pretrained MoE models. Experiments across math, coding, law, and writing show consistent improvements over TopK and prior parameter-free adaptive methods, with gains that become substantially larger under higher sparsity (up to 16.9%). These results highlight SeqTopK as a simple, efficient, and scalable routing strategy, particularly well-suited for the extreme sparsity regimes of next-generation LLMs. Code is available at this https URL. 

---
# Textual Self-attention Network: Test-Time Preference Optimization through Textual Gradient-based Attention 

**Authors**: Shibing Mo, Haoyang Ruan, Kai Wu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06682)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities, but aligning their outputs with human preferences typically requires expensive supervised fine-tuning. Recent test-time methods leverage textual feedback to overcome this, but they often critique and revise a single candidate response, lacking a principled mechanism to systematically analyze, weigh, and synthesize the strengths of multiple promising candidates. Such a mechanism is crucial because different responses may excel in distinct aspects (e.g., clarity, factual accuracy, or tone), and combining their best elements may produce a far superior outcome. This paper proposes the Textual Self-Attention Network (TSAN), a new paradigm for test-time preference optimization that requires no parameter updates. TSAN emulates self-attention entirely in natural language to overcome this gap: it analyzes multiple candidates by formatting them into textual keys and values, weighs their relevance using an LLM-based attention module, and synthesizes their strengths into a new, preference-aligned response under the guidance of the learned textual attention. This entire process operates in a textual gradient space, enabling iterative and interpretable optimization. Empirical evaluations demonstrate that with just three test-time iterations on a base SFT model, TSAN outperforms supervised models like Llama-3.1-70B-Instruct and surpasses the current state-of-the-art test-time alignment method by effectively leveraging multiple candidate solutions. 

---
# Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint 

**Authors**: Suqing Wang, Ziyang Ma, Xinyi Li, Zuchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06390)  

**Abstract**: Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models. 

---
# TimeSense:Making Large Language Models Proficient in Time-Series Analysis 

**Authors**: Zhirui Zhang, Changhua Pei, Tianyi Gao, Zhe Xie, Yibo Hao, Zhaoyang Yu, Longlong Xu, Tong Xiao, Jing Han, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2511.06344)  

**Abstract**: In the time-series domain, an increasing number of works combine text with temporal data to leverage the reasoning capabilities of large language models (LLMs) for various downstream time-series understanding tasks. This enables a single model to flexibly perform tasks that previously required specialized models for each domain. However, these methods typically rely on text labels for supervision during training, biasing the model toward textual cues while potentially neglecting the full temporal features. Such a bias can lead to outputs that contradict the underlying time-series context. To address this issue, we construct the EvalTS benchmark, comprising 10 tasks across three difficulty levels, from fundamental temporal pattern recognition to complex real-world reasoning, to evaluate models under more challenging and realistic scenarios. We also propose TimeSense, a multimodal framework that makes LLMs proficient in time-series analysis by balancing textual reasoning with a preserved temporal sense. TimeSense incorporates a Temporal Sense module that reconstructs the input time-series within the model's context, ensuring that textual reasoning is grounded in the time-series dynamics. Moreover, to enhance spatial understanding of time-series data, we explicitly incorporate coordinate-based positional embeddings, which provide each time point with spatial context and enable the model to capture structural dependencies more effectively. Experimental results demonstrate that TimeSense achieves state-of-the-art performance across multiple tasks, and it particularly outperforms existing methods on complex multi-dimensional time-series reasoning tasks. 

---
# PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization 

**Authors**: Kelun Lei, Hailong Yang, Huaitao Zhang, Xin You, Kaige Zhang, Zhongzhi Luan, Yi Liu, Depei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.06345)  

**Abstract**: Designing high-performance kernels requires expert-level tuning and a deep understanding of hardware characteristics. Recent advances in large language models (LLMs) have enabled automated kernel generation, yet most existing systems rely solely on correctness or execution time feedback, lacking the ability to reason about low-level performance bottlenecks. In this paper, we introduce PRAGMA, a profile-guided AI kernel generation framework that integrates execution feedback and fine-grained hardware profiling into the reasoning loop. PRAGMA enables LLMs to identify performance bottlenecks, preserve historical best versions, and iteratively refine code quality. We evaluate PRAGMA on KernelBench, covering GPU and CPU backends. Results show that PRAGMA consistently outperforms baseline AIKG without profiling enabled and achieves 2.81$\times$ and 2.30$\times$ averaged speedups against Torch on CPU and GPU platforms, respectively. 

---
# SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention 

**Authors**: Bohan Yu, Wei Huang, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06446)  

**Abstract**: This paper proposes SR-KI, a novel approach for integrating real-time and large-scale structured knowledge bases (KBs) into large language models (LLMs). SR-KI begins by encoding KBs into key-value pairs using a pretrained encoder, and injects them into LLMs' KV cache. Building on this representation, we employ a two-stage training paradigm: first locating a dedicated retrieval layer within the LLM, and then applying an attention-based loss at this layer to explicitly supervise attention toward relevant KB entries. Unlike traditional retrieval-augmented generation methods that rely heavily on the performance of external retrievers and multi-stage pipelines, SR-KI supports end-to-end inference by performing retrieval entirely within the models latent space. This design enables efficient compression of injected knowledge and facilitates dynamic knowledge updates. Comprehensive experiments demonstrate that SR-KI enables the integration of up to 40K KBs into a 7B LLM on a single A100 40GB GPU, and achieves strong retrieval performance, maintaining over 98% Recall@10 on the best-performing task and exceeding 88% on average across all tasks. Task performance on question answering and KB ID generation also demonstrates that SR-KI maintains strong performance while achieving up to 99.75% compression of the injected KBs. 

---
# LLM-Guided Reinforcement Learning with Representative Agents for Traffic Modeling 

**Authors**: Hanlin Sun, Jiayang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06260)  

**Abstract**: Large language models (LLMs) are increasingly used as behavioral proxies for self-interested travelers in agent-based traffic models. Although more flexible and generalizable than conventional models, the practical use of these approaches remains limited by scalability due to the cost of calling one LLM for every traveler. Moreover, it has been found that LLM agents often make opaque choices and produce unstable day-to-day dynamics. To address these challenges, we propose to model each homogeneous traveler group facing the same decision context with a single representative LLM agent who behaves like the population's average, maintaining and updating a mixed strategy over routes that coincides with the group's aggregate flow proportions. Each day, the LLM reviews the travel experience and flags routes with positive reinforcement that they hope to use more often, and an interpretable update rule then converts this judgment into strategy adjustments using a tunable (progressively decaying) step size. The representative-agent design improves scalability, while the separation of reasoning from updating clarifies the decision logic while stabilizing learning. In classic traffic assignment settings, we find that the proposed approach converges rapidly to the user equilibrium. In richer settings with income heterogeneity, multi-criteria costs, and multi-modal choices, the generated dynamics remain stable and interpretable, reproducing plausible behavioral patterns well-documented in psychology and economics, for example, the decoy effect in toll versus non-toll road selection, and higher willingness-to-pay for convenience among higher-income travelers when choosing between driving, transit, and park-and-ride options. 

---
# Decomate: Leveraging Generative Models for Co-Creative SVG Animation 

**Authors**: Jihyeon Park, Jiyoon Myung, Seone Shin, Jungki Son, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.06297)  

**Abstract**: Designers often encounter friction when animating static SVG graphics, especially when the visual structure does not match the desired level of motion detail. Existing tools typically depend on predefined groupings or require technical expertise, which limits designers' ability to experiment and iterate independently. We present Decomate, a system that enables intuitive SVG animation through natural language. Decomate leverages a multimodal large language model to restructure raw SVGs into semantically meaningful, animation-ready components. Designers can then specify motions for each component via text prompts, after which the system generates corresponding HTML/CSS/JS animations. By supporting iterative refinement through natural language interaction, Decomate integrates generative AI into creative workflows, allowing animation outcomes to be directly shaped by user intent. 

---
# Mixtures of SubExperts for Large Language Continual Learning 

**Authors**: Haeyong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06237)  

**Abstract**: Adapting Large Language Models (LLMs) to a continuous stream of tasks is a critical yet challenging endeavor. While Parameter-Efficient Fine-Tuning (PEFT) methods have become a standard for this, they face a fundamental dilemma in continual learning. Reusing a single set of PEFT parameters for new tasks often leads to catastrophic forgetting of prior knowledge. Conversely, allocating distinct parameters for each task prevents forgetting but results in a linear growth of the model's size and fails to facilitate knowledge transfer between related tasks. To overcome these limitations, we propose a novel adaptive PEFT method referred to as \textit{Mixtures of SubExperts (MoSEs)}, a novel continual learning framework designed for minimal forgetting and efficient scalability. MoSEs integrate a sparse Mixture of SubExperts into the transformer layers, governed by a task-specific routing mechanism. This architecture allows the model to isolate and protect knowledge within dedicated SubExperts, thereby minimizing parameter interference and catastrophic forgetting. Crucially, the router can adaptively select and combine previously learned sparse parameters for new tasks, enabling effective knowledge transfer while ensuring that the model's capacity grows sublinearly. We evaluate MoSEs on the comprehensive TRACE benchmark datasets. Our experiments demonstrate that MoSEs significantly outperform conventional continual learning approaches in both knowledge retention and scalability to new tasks, achieving state-of-the-art performance with substantial memory and computational savings. 

---
# Assertion-Aware Test Code Summarization with Large Language Models 

**Authors**: Anamul Haque Mollah, Ahmed Aljohani, Hyunsook Do  

**Link**: [PDF](https://arxiv.org/pdf/2511.06227)  

**Abstract**: Unit tests often lack concise summaries that convey test intent, especially in auto-generated or poorly documented codebases. Large Language Models (LLMs) offer a promising solution, but their effectiveness depends heavily on how they are prompted. Unlike generic code summarization, test-code summarization poses distinct challenges because test methods validate expected behavior through assertions rather than im- plementing functionality. This paper presents a new benchmark of 91 real-world Java test cases paired with developer-written summaries and conducts a controlled ablation study to investigate how test code-related components-such as the method under test (MUT), assertion messages, and assertion semantics-affect the performance of LLM-generated test summaries. We evaluate four code LLMs (Codex, Codestral, DeepSeek, and Qwen-Coder) across seven prompt configurations using n-gram metrics (BLEU, ROUGE-L, METEOR), semantic similarity (BERTScore), and LLM-based evaluation. Results show that prompting with as- sertion semantics improves summary quality by an average of 0.10 points (2.3%) over full MUT context (4.45 vs. 4.35) while requiring fewer input tokens. Codex and Qwen-Coder achieve the highest alignment with human-written summaries, while DeepSeek underperforms despite high lexical overlap. The replication package is publicly available at this https URL. 5281/zenodo.17067550 

---
# Overview of CHIP 2025 Shared Task 2: Discharge Medication Recommendation for Metabolic Diseases Based on Chinese Electronic Health Records 

**Authors**: Juntao Li, Haobin Yuan, Ling Luo, Tengxiao Lv, Yan Jiang, Fan Wang, Ping Zhang, Huiyi Lv, Jian Wang, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.06230)  

**Abstract**: Discharge medication recommendation plays a critical role in ensuring treatment continuity, preventing readmission, and improving long-term management for patients with chronic metabolic diseases. This paper present an overview of the CHIP 2025 Shared Task 2 competition, which aimed to develop state-of-the-art approaches for automatically recommending appro-priate discharge medications using real-world Chinese EHR data. For this task, we constructed CDrugRed, a high-quality dataset consisting of 5,894 de-identified hospitalization records from 3,190 patients in China. This task is challenging due to multi-label nature of medication recommendation, het-erogeneous clinical text, and patient-specific variability in treatment plans. A total of 526 teams registered, with 167 and 95 teams submitting valid results to the Phase A and Phase B leaderboards, respectively. The top-performing team achieved the highest overall performance on the final test set, with a Jaccard score of 0.5102, F1 score of 0.6267, demonstrating the potential of advanced large language model (LLM)-based ensemble systems. These re-sults highlight both the promise and remaining challenges of applying LLMs to medication recommendation in Chinese EHRs. The post-evaluation phase remains open at this https URL. 

---
# RAG-targeted Adversarial Attack on LLM-based Threat Detection and Mitigation Framework 

**Authors**: Seif Ikbarieh, Kshitiz Aryal, Maanak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.06212)  

**Abstract**: The rapid expansion of the Internet of Things (IoT) is reshaping communication and operational practices across industries, but it also broadens the attack surface and increases susceptibility to security breaches. Artificial Intelligence has become a valuable solution in securing IoT networks, with Large Language Models (LLMs) enabling automated attack behavior analysis and mitigation suggestion in Network Intrusion Detection Systems (NIDS). Despite advancements, the use of LLMs in such systems further expands the attack surface, putting entire networks at risk by introducing vulnerabilities such as prompt injection and data poisoning. In this work, we attack an LLM-based IoT attack analysis and mitigation framework to test its adversarial robustness. We construct an attack description dataset and use it in a targeted data poisoning attack that applies word-level, meaning-preserving perturbations to corrupt the Retrieval-Augmented Generation (RAG) knowledge base of the framework. We then compare pre-attack and post-attack mitigation responses from the target model, ChatGPT-5 Thinking, to measure the impact of the attack on model performance, using an established evaluation rubric designed for human experts and judge LLMs. Our results show that small perturbations degrade LLM performance by weakening the linkage between observed network traffic features and attack behavior, and by reducing the specificity and practicality of recommended mitigations for resource-constrained devices. 

---
# LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs 

**Authors**: Zifan He, Shengyu Ye, Rui Ma, Yang Wang, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06174)  

**Abstract**: The rapid progress of large language models (LLMs) has advanced numerous applications, yet efficient single-batch inference remains vital for on-device intelligence. While FPGAs offer fine-grained data control and high energy efficiency, recent GPU optimizations have narrowed their advantage, especially under arithmetic-based computation. To overcome this, we leverage FPGAs' abundant on-chip memory to shift LLM inference from arithmetic- to memory-based computation through table lookups. We present LUT-LLM, the first FPGA accelerator enabling 1B+ LLM inference via vector-quantized memory operations. Our analysis identifies activation-weight co-quantization as the most effective scheme, supported by (1) bandwidth-aware parallel centroid search, (2) efficient 2D table lookups, and (3) a spatial-temporal hybrid design minimizing data caching. Implemented on an AMD V80 FPGA for a customized Qwen 3 1.7B model, LUT-LLM achieves 1.66x lower latency than AMD MI210 and 1.72x higher energy efficiency than NVIDIA A100, scaling to 32B models with 2.16x efficiency gain over A100. 

---
# LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains 

**Authors**: Ibna Kowsar, Kazi F. Akhter, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2511.06161)  

**Abstract**: Transfer learning of tabular data is non-trivial due to heterogeneity in the feature space across disparate domains. The limited success of traditional deep learning in tabular knowledge transfer can be advanced by leveraging large language models (LLMs). However, the efficacy of LLMs often stagnates for mixed data types structured in tables due to the limitations of text prompts and in-context learning. We propose a lightweight transfer learning framework that fine-tunes an LLM using source tabular data and transplants the LLM's selective $key$ and $value$ projection weights into a gated feature tokenized transformer (gFTT) built for tabular data. The gFTT model with cross-domain attention is fine-tuned using target tabular data for transfer learning, eliminating the need for shared features, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of source-target data sets and 12 baselines demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and transfer learning models trained on thousands to billions of tabular samples. The proposed attention transfer demonstrates an effective solution to learning relationships between data tables using an LLM in a low-resource learning environment. The source code for the proposed method is publicly available. 

---
# Large Language Models Develop Novel Social Biases Through Adaptive Exploration 

**Authors**: Addison J. Wu, Ryan Liu, Xuechunzi Bai, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.06148)  

**Abstract**: As large language models (LLMs) are adopted into frameworks that grant them the capacity to make real decisions, it is increasingly important to ensure that they are unbiased. In this paper, we argue that the predominant approach of simply removing existing biases from models is not enough. Using a paradigm from the psychology literature, we demonstrate that LLMs can spontaneously develop novel social biases about artificial demographic groups even when no inherent differences exist. These biases result in highly stratified task allocations, which are less fair than assignments by human participants and are exacerbated by newer and larger models. In social science, emergent biases like these have been shown to result from exploration-exploitation trade-offs, where the decision-maker explores too little, allowing early observations to strongly influence impressions about entire demographic groups. To alleviate this effect, we examine a series of interventions targeting model inputs, problem structure, and explicit steering. We find that explicitly incentivizing exploration most robustly reduces stratification, highlighting the need for better multifaceted objectives to mitigate bias. These results reveal that LLMs are not merely passive mirrors of human social biases, but can actively create new ones from experience, raising urgent questions about how these systems will shape societies over time. 

---
# Stemming Hallucination in Language Models Using a Licensing Oracle 

**Authors**: Simeon Emanuilov, Richard Ackermann  

**Link**: [PDF](https://arxiv.org/pdf/2511.06073)  

**Abstract**: Language models exhibit remarkable natural language generation capabilities but remain prone to hallucinations, generating factually incorrect information despite producing syntactically coherent responses. This study introduces the Licensing Oracle, an architectural solution designed to stem hallucinations in LMs by enforcing truth constraints through formal validation against structured knowledge graphs. Unlike statistical approaches that rely on data scaling or fine-tuning, the Licensing Oracle embeds a deterministic validation step into the model's generative process, ensuring that only factually accurate claims are made. We evaluated the effectiveness of the Licensing Oracle through experiments comparing it with several state-of-the-art methods, including baseline language model generation, fine-tuning for factual recall, fine-tuning for abstention behavior, and retrieval-augmented generation (RAG). Our results demonstrate that although RAG and fine-tuning improve performance, they fail to eliminate hallucinations. In contrast, the Licensing Oracle achieved perfect abstention precision (AP = 1.0) and zero false answers (FAR-NE = 0.0), ensuring that only valid claims were generated with 89.1% accuracy in factual responses. This work shows that architectural innovations, such as the Licensing Oracle, offer a necessary and sufficient solution for hallucinations in domains with structured knowledge representations, offering guarantees that statistical methods cannot match. Although the Licensing Oracle is specifically designed to address hallucinations in fact-based domains, its framework lays the groundwork for truth-constrained generation in future AI systems, providing a new path toward reliable, epistemically grounded models. 

---
# Simulating Students with Large Language Models: A Review of Architecture, Mechanisms, and Role Modelling in Education with Generative AI 

**Authors**: Luis Marquez-Carpintero, Alberto Lopez-Sellers, Miguel Cazorla  

**Link**: [PDF](https://arxiv.org/pdf/2511.06078)  

**Abstract**: Simulated Students offer a valuable methodological framework for evaluating pedagogical approaches and modelling diverse learner profiles, tasks which are otherwise challenging to undertake systematically in real-world settings. Recent research has increasingly focused on developing such simulated agents to capture a range of learning styles, cognitive development pathways, and social behaviours. Among contemporary simulation techniques, the integration of large language models (LLMs) into educational research has emerged as a particularly versatile and scalable paradigm. LLMs afford a high degree of linguistic realism and behavioural adaptability, enabling agents to approximate cognitive processes and engage in contextually appropriate pedagogical dialogues. This paper presents a thematic review of empirical and methodological studies utilising LLMs to simulate student behaviour across educational environments. We synthesise current evidence on the capacity of LLM-based agents to emulate learner archetypes, respond to instructional inputs, and interact within multi-agent classroom scenarios. Furthermore, we examine the implications of such systems for curriculum development, instructional evaluation, and teacher training. While LLMs surpass rule-based systems in natural language generation and situational flexibility, ongoing concerns persist regarding algorithmic bias, evaluation reliability, and alignment with educational objectives. The review identifies existing technological and methodological gaps and proposes future research directions for integrating generative AI into adaptive learning systems and instructional design. 

---
# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models 

**Authors**: Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.05993)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a predominant approach for enhancing the reasoning capabilities of large language models (LLMs). However, the entropy of LLMs usually collapses during RLVR training, causing premature convergence to suboptimal local minima and hinder further performance improvement. Although various approaches have been proposed to mitigate entropy collapse, a comprehensive study of entropy in RLVR remains lacking. To address this gap, we conduct extensive experiments to investigate the entropy dynamics of LLMs trained with RLVR and analyze how model entropy correlates with response diversity, calibration, and performance across various benchmarks. Our findings reveal that the number of off-policy updates, the diversity of training data, and the clipping thresholds in the optimization objective are critical factors influencing the entropy of LLMs trained with RLVR. Moreover, we theoretically and empirically demonstrate that tokens with positive advantages are the primary contributors to entropy collapse, and that model entropy can be effectively regulated by adjusting the relative loss weights of tokens with positive and negative advantages during training. 

---
# MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference 

**Authors**: Myunghyun Rhee, Sookyung Choi, Euiseok Kim, Joonseop Sim, Youngpyo Joo, Hoshik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06010)  

**Abstract**: The escalating context length in Large Language Models (LLMs) creates a severe performance bottleneck around the Key-Value (KV) cache, whose memory-bound nature leads to significant GPU under-utilization. This paper introduces Mixture of Shared KV Attention (MoSKA), an architecture that addresses this challenge by exploiting the heterogeneity of context data. It differentiates between per-request unique and massively reused shared sequences. The core of MoSKA is a novel Shared KV Attention mechanism that transforms the attention on shared data from a series of memory-bound GEMV operations into a single, compute-bound GEMM by batching concurrent requests. This is supported by an MoE-inspired sparse attention strategy that prunes the search space and a tailored Disaggregated Infrastructure that specializes hardware for unique and shared data. This comprehensive approach demonstrates a throughput increase of up to 538.7x over baselines in workloads with high context sharing, offering a clear architectural path toward scalable LLM inference. 

---
# Kunlun Anomaly Troubleshooter: Enabling Kernel-Level Anomaly Detection and Causal Reasoning for Large Model Distributed Inference 

**Authors**: Yuyang Liu, Jingjing Cai, Jiayi Ren, Peng Zhou, Danyang Zhang, Yin Du, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.05978)  

**Abstract**: Anomaly troubleshooting for large model distributed inference (LMDI) remains a critical challenge. Resolving anomalies such as inference performance degradation or latency jitter in distributed system demands significant manual efforts from domain experts, resulting in extremely time-consuming diagnosis processes with relatively low accuracy. In this paper, we introduce Kunlun Anomaly Troubleshooter (KAT), the first anomaly troubleshooting framework tailored for LMDI. KAT addresses this problem through two core innovations. First, KAT exploits the synchronicity and consistency of GPU workers, innovatively leverages function trace data to precisely detect kernel-level anomalies and associated hardware components at nanosecond resolution. Second, KAT integrates these detection results into a domain-adapted LLM, delivering systematic causal reasoning and natural language interpretation of complex anomaly symptoms. Evaluations conducted in Alibaba Cloud Service production environment indicate that KAT achieves over 0.884 precision and 0.936 recall in anomaly detection, providing detail anomaly insights that significantly narrow down the diagnostic scope and improve both the efficiency and success rate of troubleshooting. 

---
# Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs 

**Authors**: Alina Fastowski, Bardh Prenkaj, Yuxiao Li, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2511.05919)  

**Abstract**: LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety. 

---
# NILC: Discovering New Intents with LLM-assisted Clustering 

**Authors**: Hongtao Wang, Renchi Yang, Wenqing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05913)  

**Abstract**: New intent discovery (NID) seeks to recognize both new and known intents from unlabeled user utterances, which finds prevalent use in practical dialogue systems. Existing works towards NID mainly adopt a cascaded architecture, wherein the first stage focuses on encoding the utterances into informative text embeddings beforehand, while the latter is to group similar embeddings into clusters (i.e., intents), typically by K-Means. However, such a cascaded pipeline fails to leverage the feedback from both steps for mutual refinement, and, meanwhile, the embedding-only clustering overlooks nuanced textual semantics, leading to suboptimal performance. To bridge this gap, this paper proposes NILC, a novel clustering framework specially catered for effective NID. Particularly, NILC follows an iterative workflow, in which clustering assignments are judiciously updated by carefully refining cluster centroids and text embeddings of uncertain utterances with the aid of large language models (LLMs). Specifically, NILC first taps into LLMs to create additional semantic centroids for clusters, thereby enriching the contextual semantics of the Euclidean centroids of embeddings. Moreover, LLMs are then harnessed to augment hard samples (ambiguous or terse utterances) identified from clusters via rewriting for subsequent cluster correction. Further, we inject supervision signals through non-trivial techniques seeding and soft must links for more accurate NID in the semi-supervised setting. Extensive experiments comparing NILC against multiple recent baselines under both unsupervised and semi-supervised settings showcase that NILC can achieve significant performance improvements over six benchmark datasets of diverse domains consistently. 

---
# EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph 

**Authors**: Nan Jiang, Ziyi Wang, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.05849)  

**Abstract**: Symbolic regression seeks to uncover physical laws from experimental data by searching for closed-form expressions, which is an important task in AI-driven scientific discovery. Yet the exponential growth of the search space of expression renders the task computationally challenging. A promising yet underexplored direction for reducing the effective search space and accelerating training lies in symbolic equivalence: many expressions, although syntactically different, define the same function -- for example, $\log(x_1^2x_2^3)$, $\log(x_1^2)+\log(x_2^3)$, and $2\log(x_1)+3\log(x_2)$. Existing algorithms treat such variants as distinct outputs, leading to redundant exploration and slow learning. We introduce EGG-SR, a unified framework that integrates equality graphs (e-graphs) into diverse symbolic regression algorithms, including Monte Carlo Tree Search (MCTS), deep reinforcement learning (DRL), and large language models (LLMs). EGG-SR compactly represents equivalent expressions through the proposed EGG module, enabling more efficient learning by: (1) pruning redundant subtree exploration in EGG-MCTS, (2) aggregating rewards across equivalence classes in EGG-DRL, and (3) enriching feedback prompts in EGG-LLM. Under mild assumptions, we show that embedding e-graphs tightens the regret bound of MCTS and reduces the variance of the DRL gradient estimator. Empirically, EGG-SR consistently enhances multiple baselines across challenging benchmarks, discovering equations with lower normalized mean squared error than state-of-the-art methods. Code implementation is available at: this https URL. 

---
# WAR-Re: Web API Recommendation with Semantic Reasoning 

**Authors**: Zishuo Xu, Dezhong Yao, Yao Wan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05820)  

**Abstract**: With the development of cloud computing, the number of Web APIs has increased dramatically, further intensifying the demand for efficient Web API recommendation. Despite the demonstrated success of previous Web API recommendation solutions, two critical challenges persist: 1) a fixed top-N recommendation that cannot accommodate the varying API cardinality requirements of different mashups, and 2) these methods output only ranked API lists without accompanying reasons, depriving users of understanding the recommendation. To address these challenges, we propose WAR-Re, an LLM-based model for Web API recommendation with semantic reasoning for justification. WAR-Re leverages special start and stop tokens to handle the first challenge and uses two-stage training: supervised fine-tuning and reinforcement learning via Group Relative Policy Optimization (GRPO) to enhance the model's ability in both tasks. Comprehensive experimental evaluations on the ProgrammableWeb dataset demonstrate that WAR-Re achieves a gain of up to 21.59\% over the state-of-the-art baseline model in recommendation accuracy, while consistently producing high-quality semantic reasons for recommendations. 

---
# DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning 

**Authors**: Yaxuan Wang, Chris Yuhao Liu, Quan Liu, Jinglong Pang, Wei Wei, Yujia Bao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05784)  

**Abstract**: Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios. 

---
# Lived Experience in Dialogue: Co-designing Personalization in Large Language Models to Support Youth Mental Well-being 

**Authors**: Kathleen W. Guan, Sarthak Giri, Mohammed Amara, Bernard J. Jansen, Enrico Liscio, Milena Esherick, Mohammed Al Owayyed, Ausrine Ratkute, Gayane Sedrakyan, Mark de Reuver, Joao Fernando Ferreira Goncalves, Caroline A. Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2511.05769)  

**Abstract**: Youth increasingly turn to large language models (LLMs) for mental well-being support, yet current personalization in LLMs can overlook the heterogeneous lived experiences shaping their needs. We conducted a participatory study with youth, parents, and youth care workers (N=38), using co-created youth personas as scaffolds, to elicit community perspectives on how LLMs can facilitate more meaningful personalization to support youth mental well-being. Analysis identified three themes: person-centered contextualization responsive to momentary needs, explicit boundaries around scope and offline referral, and dialogic scaffolding for reflection and autonomy. We mapped these themes to persuasive design features for task suggestions, social facilitation, and system trustworthiness, and created corresponding dialogue extracts to guide LLM fine-tuning. Our findings demonstrate how lived experience can be operationalized to inform design features in LLMs, which can enhance the alignment of LLM-based interventions with the realities of youth and their communities, contributing to more effectively personalized digital well-being tools. 

---
# AdvisingWise: Supporting Academic Advising in Higher Educations Through a Human-in-the-Loop Multi-Agent Framework 

**Authors**: Wendan Jiang, Shiyuan Wang, Hiba Eltigani, Rukhshan Haroon, Abdullah Bin Faisal, Fahad Dogar  

**Link**: [PDF](https://arxiv.org/pdf/2511.05706)  

**Abstract**: Academic advising is critical to student success in higher education, yet high student-to-advisor ratios limit advisors' capacity to provide timely support, particularly during peak periods. Recent advances in Large Language Models (LLMs) present opportunities to enhance the advising process. We present AdvisingWise, a multi-agent system that automates time-consuming tasks, such as information retrieval and response drafting, while preserving human oversight. AdvisingWise leverages authoritative institutional resources and adaptively prompts students about their academic backgrounds to generate reliable, personalized responses. All system responses undergo human advisor validation before delivery to students. We evaluate AdvisingWise through a mixed-methods approach: (1) expert evaluation on responses of 20 sample queries, (2) LLM-as-a-judge evaluation of the information retrieval strategy, and (3) a user study with 8 academic advisors to assess the system's practical utility. Our evaluation shows that AdvisingWise produces accurate, personalized responses. Advisors reported increasingly positive perceptions after using AdvisingWise, as their initial concerns about reliability and personalization diminished. We conclude by discussing the implications of human-AI synergy on the practice of academic advising. 

---
# LLMs as Packagers of HPC Software 

**Authors**: Caetano Melone, Daniel Nichols, Konstantinos Parasyris, Todd Gamblin, Harshitha Menon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05626)  

**Abstract**: High performance computing (HPC) software ecosystems are inherently heterogeneous, comprising scientific applications that depend on hundreds of external packages, each with distinct build systems, options, and dependency constraints. Tools such as Spack automate dependency resolution and environment management, but their effectiveness relies on manually written build recipes. As these ecosystems grow, maintaining existing specifications and creating new ones becomes increasingly labor-intensive. While large language models (LLMs) have shown promise in code generation, automatically producing correct and maintainable Spack recipes remains a significant challenge. We present a systematic analysis of how LLMs and context-augmentation methods can assist in the generation of Spack recipes. To this end, we introduce SpackIt, an end-to-end framework that combines repository analysis, retrieval of relevant examples, and iterative refinement through diagnostic feedback. We apply SpackIt to a representative subset of 308 open-source HPC packages to assess its effectiveness and limitations. Our results show that SpackIt increases installation success from 20% in a zero-shot setting to over 80% in its best configuration, demonstrating the value of retrieval and structured feedback for reliable package synthesis. 

---
# Assessing the Reliability of Large Language Models in the Bengali Legal Context: A Comparative Evaluation Using LLM-as-Judge and Legal Experts 

**Authors**: Sabik Aftahee, A.F.M. Farhad, Arpita Mallik, Ratnajit Dhar, Jawadul Karim, Nahiyan Bin Noor, Ishmam Ahmed Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2511.05627)  

**Abstract**: Accessing legal help in Bangladesh is hard. People face high fees, complex legal language, a shortage of lawyers, and millions of unresolved court cases. Generative AI models like OpenAI GPT-4.1 Mini, Gemini 2.0 Flash, Meta Llama 3 70B, and DeepSeek R1 could potentially democratize legal assistance by providing quick and affordable legal advice. In this study, we collected 250 authentic legal questions from the Facebook group "Know Your Rights," where verified legal experts regularly provide authoritative answers. These questions were subsequently submitted to four four advanced AI models and responses were generated using a consistent, standardized prompt. A comprehensive dual evaluation framework was employed, in which a state-of-the-art LLM model served as a judge, assessing each AI-generated response across four critical dimensions: factual accuracy, legal appropriateness, completeness, and clarity. Following this, the same set of questions was evaluated by three licensed Bangladeshi legal professionals according to the same criteria. In addition, automated evaluation metrics, including BLEU scores, were applied to assess response similarity. Our findings reveal a complex landscape where AI models frequently generate high-quality, well-structured legal responses but also produce dangerous misinformation, including fabricated case citations, incorrect legal procedures, and potentially harmful advice. These results underscore the critical need for rigorous expert validation and comprehensive safeguards before AI systems can be safely deployed for legal consultation in Bangladesh. 

---
# Optimizing Diversity and Quality through Base-Aligned Model Collaboration 

**Authors**: Yichen Wang, Chenghao Yang, Tenghao Huang, Muhao Chen, Jonathan May, Mina Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05650)  

**Abstract**: Alignment has greatly improved large language models (LLMs)' output quality at the cost of diversity, yielding highly similar outputs across generations. We propose Base-Aligned Model Collaboration (BACo), an inference-time token-level model collaboration framework that dynamically combines a base LLM with its aligned counterpart to optimize diversity and quality. Inspired by prior work (Fei et al., 2025), BACo employs routing strategies that determine, at each token, from which model to decode based on next-token prediction uncertainty and predicted contents' semantic role. Prior diversity-promoting methods, such as retraining, prompt engineering, and multi-sampling methods, improve diversity but often degrade quality or require costly decoding or post-training. In contrast, BACo achieves both high diversity and quality post hoc within a single pass, while offering strong controllability. We explore a family of routing strategies, across three open-ended generation tasks and 13 metrics covering diversity and quality, BACo consistently surpasses state-of-the-art inference-time baselines. With our best router, BACo achieves a 21.3% joint improvement in diversity and quality. Human evaluations also mirror these improvements. The results suggest that collaboration between base and aligned models can optimize and control diversity and quality. 

---
# Automated Invoice Data Extraction: Using LLM and OCR 

**Authors**: Advait Thakur, Khushi Khanchandani, Akshita Shetty, Chaitravi Reddy, Ritisa Behera  

**Link**: [PDF](https://arxiv.org/pdf/2511.05547)  

**Abstract**: Conventional Optical Character Recognition (OCR) systems are challenged by variant invoice layouts, handwritten text, and low- quality scans, which are often caused by strong template dependencies that restrict their flexibility across different document structures and layouts. Newer solutions utilize advanced deep learning models such as Convolutional Neural Networks (CNN) as well as Transformers, and domain-specific models for better layout analysis and accuracy across various sections over varied document types. Large Language Models (LLMs) have revolutionized extraction pipelines at their core with sophisticated entity recognition and semantic comprehension to support complex contextual relationship mapping without direct programming specification. Visual Named Entity Recognition (NER) capabilities permit extraction from invoice images with greater contextual sensitivity and much higher accuracy rates than older approaches. Existing industry best practices utilize hybrid architectures that blend OCR technology and LLM for maximum scalability and minimal human intervention. This work introduces a holistic Artificial Intelligence (AI) platform combining OCR, deep learning, LLMs, and graph analytics to achieve unprecedented extraction quality and consistency. 

---
# ConnectomeBench: Can LLMs Proofread the Connectome? 

**Authors**: Jeff Brown, Andrew Kirjner Annika Vivekananthan, Ed Boyden  

**Link**: [PDF](https://arxiv.org/pdf/2511.05542)  

**Abstract**: Connectomics - the mapping of neural connections in an organism's brain - currently requires extraordinary human effort to proofread the data collected from imaging and machine-learning assisted segmentation. With the growing excitement around using AI agents to automate important scientific tasks, we explore whether current AI systems can perform multiple tasks necessary for data proofreading. We introduce ConnectomeBench, a multimodal benchmark evaluating large language model (LLM) capabilities in three critical proofreading tasks: segment type identification, split error correction, and merge error detection. Using expert annotated data from two large open-source datasets - a cubic millimeter of mouse visual cortex and the complete Drosophila brain - we evaluate proprietary multimodal LLMs including Claude 3.7/4 Sonnet, o4-mini, GPT-4.1, GPT-4o, as well as open source models like InternVL-3 and NVLM. Our results demonstrate that current models achieve surprisingly high performance in segment identification (52-82% balanced accuracy vs. 20-25% chance) and binary/multiple choice split error correction (75-85% accuracy vs. 50% chance) while generally struggling on merge error identification tasks. Overall, while the best models still lag behind expert performance, they demonstrate promising capabilities that could eventually enable them to augment and potentially replace human proofreading in connectomics. Project page: this https URL and Dataset this https URL 

---
# Retracing the Past: LLMs Emit Training Data When They Get Lost 

**Authors**: Myeongseob Ko, Nikhil Reddy Billa, Adam Nguyen, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.05518)  

**Abstract**: The memorization of training data in large language models (LLMs) poses significant privacy and copyright concerns. Existing data extraction methods, particularly heuristic-based divergence attacks, often exhibit limited success and offer limited insight into the fundamental drivers of memorization leakage. This paper introduces Confusion-Inducing Attacks (CIA), a principled framework for extracting memorized data by systematically maximizing model uncertainty. We empirically demonstrate that the emission of memorized text during divergence is preceded by a sustained spike in token-level prediction entropy. CIA leverages this insight by optimizing input snippets to deliberately induce this consecutive high-entropy state. For aligned LLMs, we further propose Mismatched Supervised Fine-tuning (SFT) to simultaneously weaken their alignment and induce targeted confusion, thereby increasing susceptibility to our attacks. Experiments on various unaligned and aligned LLMs demonstrate that our proposed attacks outperform existing baselines in extracting verbatim and near-verbatim training data without requiring prior knowledge of the training data. Our findings highlight persistent memorization risks across various LLMs and offer a more systematic method for assessing these vulnerabilities. 

---
# Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS 

**Authors**: Varun Rajesh, Om Jodhpurkar, Pooja Anbuselvan, Mantinder Singh, Ashok Jallepali, Shantanu Godbole, Pradeep Kumar Sharma, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.05502)  

**Abstract**: We present a systematic, empirical evaluation of five local large language model (LLM) runtimes on Apple Silicon: MLX, MLC-LLM, this http URL, Ollama, and PyTorch MPS. Experiments were conducted on a Mac Studio equipped with an M2 Ultra processor and 192 GB of unified memory. Using the Qwen-2.5 model family across prompts ranging from a few hundred to 100,000 tokens, we measure time-to-first-token (TTFT), steady-state throughput, latency percentiles, long-context behavior (key-value and prompt caching), quantization support, streaming performance, batching and concurrency behavior, and deployment complexity.
Under our settings, MLX achieves the highest sustained generation throughput, while MLC-LLM delivers consistently lower TTFT for moderate prompt sizes and offers stronger out-of-the-box inference features. this http URL is highly efficient for lightweight single-stream use, Ollama emphasizes developer ergonomics but lags in throughput and TTFT, and PyTorch MPS remains limited by memory constraints on large models and long contexts.
All frameworks execute fully on-device with no telemetry, ensuring strong privacy guarantees. We release scripts, logs, and plots to reproduce all results. Our analysis clarifies the design trade-offs in Apple-centric LLM deployments and provides evidence-based recommendations for interactive and long-context processing. Although Apple Silicon inference frameworks still trail NVIDIA GPU-based systems such as vLLM in absolute performance, they are rapidly maturing into viable, production-grade solutions for private, on-device LLM inference. 

---
# AI Brown and AI Koditex: LLM-Generated Corpora Comparable to Traditional Corpora of English and Czech Texts 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.22996)  

**Abstract**: This article presents two corpora of English and Czech texts generated with large language models (LLMs). The motivation is to create a resource for comparing human-written texts with LLM-generated text linguistically. Emphasis was placed on ensuring these resources are multi-genre and rich in terms of topics, authors, and text types, while maintaining comparability with existing human-created corpora. These generated corpora replicate reference human corpora: BE21 by Paul Baker, which is a modern version of the original Brown Corpus, and Koditex corpus that also follows the Brown Corpus tradition but in Czech. The new corpora were generated using models from OpenAI, Anthropic, Alphabet, Meta, and DeepSeek, ranging from GPT-3 (davinci-002) to GPT-4.5, and are tagged according to the Universal Dependencies standard (i.e., they are tokenized, lemmatized, and morphologically and syntactically annotated). The subcorpus size varies according to the model used (the English part contains on average 864k tokens per model, 27M tokens altogether, the Czech partcontains on average 768k tokens per model, 21.5M tokens altogether). The corpora are freely available for download under the CC BY 4.0 license (the annotated data are under CC BY-NC-SA 4.0 licence) and are also accessible through the search interface of the Czech National Corpus. 

---
# ConvFill: Model Collaboration for Responsive Conversational Voice Agents 

**Authors**: Vidya Srinivas, Zachary Englhardt, Maximus Powers, Shwetak Patel, Vikram Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2511.07397)  

**Abstract**: Deploying conversational voice agents with large language models faces a critical challenge: cloud-based foundation models provide deep reasoning and domain knowledge but introduce latency that disrupts natural conversation, while on-device models respond immediately but lack sophistication. We propose conversational infill, a task where a lightweight on-device model generates contextually appropriate dialogue while seamlessly incorporating streaming knowledge from a powerful backend model. This approach decouples response latency from model capability, enabling systems that feel responsive while accessing the full power of large-scale models. We present ConvFill, a 360M parameter model trained on synthetic multi-domain conversations. Evaluation across multiple backend models shows that conversational infill can be successfully learned, with ConvFill achieving accuracy improvements of 36-42% over standalone small models of the same size while consistently retaining sub-200ms response latencies. Our results demonstrate the promise of this approach for building on-device conversational agents that are both immediately responsive and knowledgeable. 

---
# Retriv at BLP-2025 Task 2: Test-Driven Feedback-Guided Framework for Bangla-to-Python Code Generation 

**Authors**: K M Nafi Asib, Sourav Saha, Mohammed Moshiul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2511.07382)  

**Abstract**: Large Language Models (LLMs) have advanced the automated generation of code from natural language prompts. However, low-resource languages (LRLs) like Bangla remain underrepresented due to the limited availability of instruction-to-code datasets and evaluation benchmarks. To address this, the BLP Workshop at IJCNLP-AACL 2025 introduced a shared task on "Code Generation in Bangla". In this work, we propose a method that combines instruction prompting with a test-driven, feedback-guided iterative refinement process using a fine-tuned Qwen2.5-14B model. The model generates code from Bangla instructions, tests it against unit tests, and iteratively refines any failing outputs through three evaluation passes, using test feedback to guide each step. This approach helped our team "Retriv" to secure 2nd place in the shared task with a Pass@1 score of 0.934. The analysis highlights challenges in Bangla instruction understanding and Python code generation, emphasizing the need for targeted methods in LRLs. We made experimental scripts publicly available for the community. 

---
# Who Is the Story About? Protagonist Entity Recognition in News 

**Authors**: Jorge Gabín, M. Eduardo Ares, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07296)  

**Abstract**: News articles often reference numerous organizations, but traditional Named Entity Recognition (NER) treats all mentions equally, obscuring which entities genuinely drive the narrative. This limits downstream tasks that rely on understanding event salience, influence, or narrative focus. We introduce Protagonist Entity Recognition (PER), a task that identifies the organizations that anchor a news story and shape its main developments. To validate PER, we compare he predictions of Large Language Models (LLMs) against annotations from four expert annotators over a gold corpus, establishing both inter-annotator consistency and human-LLM agreement. Leveraging these findings, we use state-of-the-art LLMs to automatically label large-scale news collections through NER-guided prompting, generating scalable, high-quality supervision. We then evaluate whether other LLMs, given reduced context and without explicit candidate guidance, can still infer the correct protagonists. Our results demonstrate that PER is a feasible and meaningful extension to narrative-centered information extraction, and that guided LLMs can approximate human judgments of narrative importance at scale. 

---
# EMODIS: A Benchmark for Context-Dependent Emoji Disambiguation in Large Language Models 

**Authors**: Jiacheng Huang, Ning Yu, Xiaoyin Yi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07193)  

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world communication settings, yet their ability to resolve context-dependent ambiguity remains underexplored. In this work, we present EMODIS, a new benchmark for evaluating LLMs' capacity to interpret ambiguous emoji expressions under minimal but contrastive textual contexts. Each instance in EMODIS comprises an ambiguous sentence containing an emoji, two distinct disambiguating contexts that lead to divergent interpretations, and a specific question that requires contextual reasoning. We evaluate both open-source and API-based LLMs, and find that even the strongest models frequently fail to distinguish meanings when only subtle contextual cues are present. Further analysis reveals systematic biases toward dominant interpretations and limited sensitivity to pragmatic contrast. EMODIS provides a rigorous testbed for assessing contextual disambiguation, and highlights the gap in semantic reasoning between humans and LLMs. 

---
# TCM-Eval: An Expert-Level Dynamic and Extensible Benchmark for Traditional Chinese Medicine 

**Authors**: Zihao Cheng, Yuheng Lu, Huaiqian Ye, Zeming Liu, Minqi Wang, Jingjing Liu, Zihan Li, Wei Fan, Yuanfang Guo, Ruiji Fu, Shifeng She, Gang Wang, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07148)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in modern medicine, yet their application in Traditional Chinese Medicine (TCM) remains severely limited by the absence of standardized benchmarks and the scarcity of high-quality training data. To address these challenges, we introduce TCM-Eval, the first dynamic and extensible benchmark for TCM, meticulously curated from national medical licensing examinations and validated by TCM experts. Furthermore, we construct a large-scale training corpus and propose Self-Iterative Chain-of-Thought Enhancement (SI-CoTE) to autonomously enrich question-answer pairs with validated reasoning chains through rejection sampling, establishing a virtuous cycle of data and model co-evolution. Using this enriched training data, we develop ZhiMingTang (ZMT), a state-of-the-art LLM specifically designed for TCM, which significantly exceeds the passing threshold for human practitioners. To encourage future research and development, we release a public leaderboard, fostering community engagement and continuous improvement. 

---
# EmoBang: Detecting Emotion From Bengali Texts 

**Authors**: Abdullah Al Maruf, Aditi Golder, Zakaria Masud Jiyad, Abdullah Al Numan, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2511.07077)  

**Abstract**: Emotion detection from text seeks to identify an individual's emotional or mental state - positive, negative, or neutral - based on linguistic cues. While significant progress has been made for English and other high-resource languages, Bengali remains underexplored despite being the world's fourth most spoken language. The lack of large, standardized datasets classifies Bengali as a low-resource language for emotion detection. Existing studies mainly employ classical machine learning models with traditional feature engineering, yielding limited performance. In this paper, we introduce a new Bengali emotion dataset annotated across eight emotion categories and propose two models for automatic emotion detection: (i) a hybrid Convolutional Recurrent Neural Network (CRNN) model (EmoBangHybrid) and (ii) an AdaBoost-Bidirectional Encoder Representations from Transformers (BERT) ensemble model (EmoBangEnsemble). Additionally, we evaluate six baseline models with five feature engineering techniques and assess zero-shot and few-shot large language models (LLMs) on the dataset. To the best of our knowledge, this is the first comprehensive benchmark for Bengali emotion detection. Experimental results show that EmoBangH and EmoBangE achieve accuracies of 92.86% and 93.69%, respectively, outperforming existing methods and establishing strong baselines for future research. 

---
# Importance-Aware Data Selection for Efficient LLM Instruction Tuning 

**Authors**: Tingyu Jiang, Shen Li, Yiyao Song, Lan Zhang, Hualei Zhu, Yuan Zhao, Xiaohang Xu, Kenjiro Taura, Hao Henry Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07074)  

**Abstract**: Instruction tuning plays a critical role in enhancing the performance and efficiency of Large Language Models (LLMs). Its success depends not only on the quality of the instruction data but also on the inherent capabilities of the LLM itself. Some studies suggest that even a small amount of high-quality data can achieve instruction fine-tuning results that are on par with, or even exceed, those from using a full-scale dataset. However, rather than focusing solely on calculating data quality scores to evaluate instruction data, there is a growing need to select high-quality data that maximally enhances the performance of instruction tuning for a given LLM. In this paper, we propose the Model Instruction Weakness Value (MIWV) as a novel metric to quantify the importance of instruction data in enhancing model's capabilities. The MIWV metric is derived from the discrepancies in the model's responses when using In-Context Learning (ICL), helping identify the most beneficial data for enhancing instruction tuning performance. Our experimental results demonstrate that selecting only the top 1\% of data based on MIWV can outperform training on the full dataset. Furthermore, this approach extends beyond existing research that focuses on data quality scoring for data selection, offering strong empirical evidence supporting the effectiveness of our proposed method. 

---
# EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers 

**Authors**: Yilin Jiang, Mingzi Zhang, Xuanyu Yin, Sheng Jin, Suyu Lu, Zuocan Ying, Zengyi Yu, Xiangjie Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.06890)  

**Abstract**: Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See this https URL for Materials. 

---
# HLPD: Aligning LLMs to Human Language Preference for Machine-Revised Text Detection 

**Authors**: Fangqi Dai, Xingjian Jiang, Zizhuang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.06942)  

**Abstract**: To prevent misinformation and social issues arising from trustworthy-looking content generated by LLMs, it is crucial to develop efficient and reliable methods for identifying the source of texts. Previous approaches have demonstrated exceptional performance in detecting texts fully generated by LLMs. However, these methods struggle when confronting more advanced LLM output or text with adversarial multi-task machine revision, especially in the black-box setting, where the generating model is unknown. To address this challenge, grounded in the hypothesis that human writing possesses distinctive stylistic patterns, we propose Human Language Preference Detection (HLPD). HLPD employs a reward-based alignment process, Human Language Preference Optimization (HLPO), to shift the scoring model's token distribution toward human-like writing, making the model more sensitive to human writing, therefore enhancing the identification of machine-revised text. We test HLPD in an adversarial multi-task evaluation framework that leverages a five-dimensional prompt generator and multiple advanced LLMs to create diverse revision scenarios. When detecting texts revised by GPT-series models, HLPD achieves a 15.11% relative improvement in AUROC over ImBD, surpassing Fast-DetectGPT by 45.56%. When evaluated on texts generated by advanced LLMs, HLPD achieves the highest average AUROC, exceeding ImBD by 5.53% and Fast-DetectGPT by 34.14%. Code will be made available at this https URL. 

---
# Evaluating LLMs for Anxiety, Depression, and Stress Detection Evaluating Large Language Models for Anxiety, Depression, and Stress Detection: Insights into Prompting Strategies and Synthetic Data 

**Authors**: Mihael Arcan, David-Paul Niland  

**Link**: [PDF](https://arxiv.org/pdf/2511.07044)  

**Abstract**: Mental health disorders affect over one-fifth of adults globally, yet detecting such conditions from text remains challenging due to the subtle and varied nature of symptom expression. This study evaluates multiple approaches for mental health detection, comparing Large Language Models (LLMs) such as Llama and GPT with classical machine learning and transformer-based architectures including BERT, XLNet, and Distil-RoBERTa. Using the DAIC-WOZ dataset of clinical interviews, we fine-tuned models for anxiety, depression, and stress classification and applied synthetic data generation to mitigate class imbalance. Results show that Distil-RoBERTa achieved the highest F1 score (0.883) for GAD-2, while XLNet outperformed others on PHQ tasks (F1 up to 0.891). For stress detection, a zero-shot synthetic approach (SD+Zero-Shot-Basic) reached an F1 of 0.884 and ROC AUC of 0.886. Findings demonstrate the effectiveness of transformer-based models and highlight the value of synthetic data in improving recall and generalization. However, careful calibration is required to prevent precision loss. Overall, this work emphasizes the potential of combining advanced language models and data augmentation to enhance automated mental health assessment from text. 

---
# Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs 

**Authors**: Yingfeng Luo, Ziqiang Xu, Yuxuan Ouyang, Murun Yang, Dingyang Lin, Kaiyan Chang, Tong Zheng, Bei Li, Peinan Feng, Quan Du, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07003)  

**Abstract**: Large language models have significantly advanced Multilingual Machine Translation (MMT), yet the broad language coverage, consistent translation quality, and English-centric bias remain open challenges. To address these challenges, we introduce \textbf{LMT}, a suite of \textbf{L}arge-scale \textbf{M}ultilingual \textbf{T}ranslation models centered on both Chinese and English, covering 60 languages and 234 translation directions. During development, we identify a previously overlooked phenomenon of \textbf{directional degeneration}, where symmetric multi-way fine-tuning data overemphasize reverse directions (X $\to$ En/Zh), leading to excessive many-to-one mappings and degraded translation quality. We propose \textbf{Strategic Downsampling}, a simple yet effective method to mitigate this degeneration. In addition, we design \textbf{Parallel Multilingual Prompting (PMP)}, which leverages typologically related auxiliary languages to enhance cross-lingual transfer. Through rigorous data curation and refined adaptation strategies, LMT achieves SOTA performance among models of comparable language coverage, with our 4B model (LMT-60-4B) surpassing the much larger Aya-101-13B and NLLB-54B models by a substantial margin. We release LMT in four sizes (0.6B/1.7B/4B/8B) to catalyze future research and provide strong baselines for inclusive, scalable, and high-quality MMT \footnote{\href{this https URL}{this https URL}}. 

---
# Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights 

**Authors**: Hyunjae Kim, Jiwoong Sohn, Aidan Gilson, Nicholas Cochran-Caggiano, Serina Applebaum, Heeju Jin, Seihee Park, Yujin Park, Jiyeong Park, Seoyoung Choi, Brittany Alexandra Herrera Contreras, Thomas Huang, Jaehoon Yun, Ethan F. Wei, Roy Jiang, Leah Colucci, Eric Lai, Amisha Dave, Tuo Guo, Maxwell B. Singer, Yonghoe Koo, Ron A. Adelman, James Zou, Andrew Taylor, Arman Cohan, Hua Xu, Qingyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06738)  

**Abstract**: Large language models (LLMs) are transforming the landscape of medicine, yet two fundamental challenges persist: keeping up with rapidly evolving medical knowledge and providing verifiable, evidence-grounded reasoning. Retrieval-augmented generation (RAG) has been widely adopted to address these limitations by supplementing model outputs with retrieved evidence. However, whether RAG reliably achieves these goals remains unclear. Here, we present the most comprehensive expert evaluation of RAG in medicine to date. Eighteen medical experts contributed a total of 80,502 annotations, assessing 800 model outputs generated by GPT-4o and Llama-3.1-8B across 200 real-world patient and USMLE-style queries. We systematically decomposed the RAG pipeline into three components: (i) evidence retrieval (relevance of retrieved passages), (ii) evidence selection (accuracy of evidence usage), and (iii) response generation (factuality and completeness of outputs). Contrary to expectation, standard RAG often degraded performance: only 22% of top-16 passages were relevant, evidence selection remained weak (precision 41-43%, recall 27-49%), and factuality and completeness dropped by up to 6% and 5%, respectively, compared with non-RAG variants. Retrieval and evidence selection remain key failure points for the model, contributing to the overall performance drop. We further show that simple yet effective strategies, including evidence filtering and query reformulation, substantially mitigate these issues, improving performance on MedMCQA and MedXpertQA by up to 12% and 8.2%, respectively. These findings call for re-examining RAG's role in medicine and highlight the importance of stage-aware evaluation and deliberate system design for reliable medical LLM applications. 

---
# Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation 

**Authors**: Keunhyeung Park, Seunguk Yu, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06680)  

**Abstract**: Standard-to-dialect machine translation remains challenging due to a persistent dialect gap in large language models and evaluation distortions inherent in n-gram metrics, which favor source copying over authentic dialect translation. In this paper, we propose the dialect refinement (DIA-REFINE) framework, which guides LLMs toward faithful target dialect outputs through an iterative loop of translation, verification, and feedback using external dialect classifiers. To address the limitations of n-gram-based metrics, we introduce the dialect fidelity score (DFS) to quantify linguistic shift and the target dialect ratio (TDR) to measure the success of dialect translation. Experiments on Korean dialects across zero-shot and in-context learning baselines demonstrate that DIA-REFINE consistently enhances dialect fidelity. The proposed metrics distinguish between False Success cases, where high n-gram scores obscure failures in dialectal translation, and True Attempt cases, where genuine attempts at dialectal translation yield low n-gram scores. We also observed that models exhibit varying degrees of responsiveness to the framework, and that integrating in-context examples further improves the translation of dialectal expressions. Our work establishes a robust framework for goal-directed, inclusive dialect translation, providing both rigorous evaluation and critical insights into model performance. 

---
# MedVoiceBias: A Controlled Study of Audio LLM Behavior in Clinical Decision-Making 

**Authors**: Zhi Rui Tam, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.06592)  

**Abstract**: As large language models transition from text-based interfaces to audio interactions in clinical settings, they might introduce new vulnerabilities through paralinguistic cues in audio. We evaluated these models on 170 clinical cases, each synthesized into speech from 36 distinct voice profiles spanning variations in age, gender, and emotion. Our findings reveal a severe modality bias: surgical recommendations for audio inputs varied by as much as 35% compared to identical text-based inputs, with one model providing 80% fewer recommendations. Further analysis uncovered age disparities of up to 12% between young and elderly voices, which persisted in most models despite chain-of-thought prompting. While explicit reasoning successfully eliminated gender bias, the impact of emotion was not detected due to poor recognition performance. These results demonstrate that audio LLMs are susceptible to making clinical decisions based on a patient's voice characteristics rather than medical evidence, a flaw that risks perpetuating healthcare disparities. We conclude that bias-aware architectures are essential and urgently needed before the clinical deployment of these models. 

---
# Better Datasets Start From RefineLab: Automatic Optimization for High-Quality Dataset Refinement 

**Authors**: Xiaonan Luo, Yue Huang, Ping He, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06530)  

**Abstract**: High-quality Question-Answer (QA) datasets are foundational for reliable Large Language Model (LLM) evaluation, yet even expert-crafted datasets exhibit persistent gaps in domain coverage, misaligned difficulty distributions, and factual inconsistencies. The recent surge in generative model-powered datasets has compounded these quality challenges. In this work, we introduce RefineLab, the first LLM-driven framework that automatically refines raw QA textual data into high-quality datasets under a controllable token-budget constraint. RefineLab takes a set of target quality attributes (such as coverage and difficulty balance) as refinement objectives, and performs selective edits within a predefined token budget to ensure practicality and efficiency. In essence, RefineLab addresses a constrained optimization problem: improving the quality of QA samples as much as possible while respecting resource limitations. With a set of available refinement operations (e.g., rephrasing, distractor replacement), RefineLab takes as input the original dataset, a specified set of target quality dimensions, and a token budget, and determines which refinement operations should be applied to each QA sample. This process is guided by an assignment module that selects optimal refinement strategies to maximize overall dataset quality while adhering to the budget constraint. Experiments demonstrate that RefineLab consistently narrows divergence from expert datasets across coverage, difficulty alignment, factual fidelity, and distractor quality. RefineLab pioneers a scalable, customizable path to reproducible dataset design, with broad implications for LLM evaluation. 

---
# How Well Do LLMs Understand Drug Mechanisms? A Knowledge + Reasoning Evaluation Dataset 

**Authors**: Sunil Mohan, Theofanis Karaletsos  

**Link**: [PDF](https://arxiv.org/pdf/2511.06418)  

**Abstract**: Two scientific fields showing increasing interest in pre-trained large language models (LLMs) are drug development / repurposing, and personalized medicine. For both, LLMs have to demonstrate factual knowledge as well as a deep understanding of drug mechanisms, so they can recall and reason about relevant knowledge in novel situations. Drug mechanisms of action are described as a series of interactions between biomedical entities, which interlink into one or more chains directed from the drug to the targeted disease. Composing the effects of the interactions in a candidate chain leads to an inference about whether the drug might be useful or not for that disease. We introduce a dataset that evaluates LLMs on both factual knowledge of known mechanisms, and their ability to reason about them under novel situations, presented as counterfactuals that the models are unlikely to have seen during training. Using this dataset, we show that o4-mini outperforms the 4o, o3, and o3-mini models from OpenAI, and the recent small Qwen3-4B-thinking model closely matches o4-mini's performance, even outperforming it in some cases. We demonstrate that the open world setting for reasoning tasks, which requires the model to recall relevant knowledge, is more challenging than the closed world setting where the needed factual knowledge is provided. We also show that counterfactuals affecting internal links in the reasoning chain present a much harder task than those affecting a link from the drug mentioned in the prompt. 

---
# Dutch Metaphor Extraction from Cancer Patients' Interviews and Forum Data using LLMs and Human in the Loop 

**Authors**: Lifeng Han, David Lindevelt, Sander Puts, Erik van Mulligen, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2511.06427)  

**Abstract**: Metaphors and metaphorical language (MLs) play an important role in healthcare communication between clinicians, patients, and patients' family members. In this work, we focus on Dutch language data from cancer patients. We extract metaphors used by patients using two data sources: (1) cancer patient storytelling interview data and (2) online forum data, including patients' posts, comments, and questions to professionals. We investigate how current state-of-the-art large language models (LLMs) perform on this task by exploring different prompting strategies such as chain of thought reasoning, few-shot learning, and self-prompting. With a human-in-the-loop setup, we verify the extracted metaphors and compile the outputs into a corpus named this http URL. We believe the extracted metaphors can support better patient care, for example shared decision making, improved communication between patients and clinicians, and enhanced patient health literacy. They can also inform the design of personalized care pathways. We share prompts and related resources at this https URL 

---
# Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning 

**Authors**: Sangmook Lee, Dohyung Kim, Hyukhun Koh, Nakyeong Yang, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2511.06190)  

**Abstract**: Recent advances in Large Language Models (LLMs) - particularly model scaling and test-time techniques - have greatly enhanced the reasoning capabilities of language models at the expense of higher inference costs. To lower inference costs, prior works train router models or deferral mechanisms that allocate easy queries to a small, efficient model, while forwarding harder queries to larger, more expensive models. However, these trained router models often lack robustness under domain shifts and require expensive data synthesis techniques such as Monte Carlo rollouts to obtain sufficient ground-truth routing labels for training. In this work, we propose Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning (STEER), a domain-agnostic framework that performs fine-grained, step-level routing between smaller and larger LLMs without utilizing external models. STEER leverages confidence scores from the smaller model's logits prior to generating a reasoning step, so that the large model is invoked only when necessary. Extensive evaluations using different LLMs on a diverse set of challenging benchmarks across multiple domains such as Mathematical Reasoning, Multi-Hop QA, and Planning tasks indicate that STEER achieves competitive or enhanced accuracy while reducing inference costs (up to +20% accuracy with 48% less FLOPs compared to solely using the larger model on AIME), outperforming baselines that rely on trained external modules. Our results establish model-internal confidence as a robust, domain-agnostic signal for model routing, offering a scalable pathway for efficient LLM deployment. 

---
# SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization 

**Authors**: Yue Huang, Xiangqi Wang, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06222)  

**Abstract**: In high-stakes scenarios-such as self-harm, legal, or medical queries-LLMs must be both trustworthy and helpful. However, these goals often conflict. We propose priority alignment, a new alignment paradigm that enforces a strict "trustworthy-before-helpful" ordering: optimization of helpfulness is conditioned on first meeting trustworthy thresholds (e.g., harmlessness or honesty). To realize this, we introduce Self-Priority Alignment (SPA)-a fully unsupervised framework that generates diverse responses, self-evaluates them and refines them by the model itself, and applies dual-criterion denoising to remove inconsistency and control variance. From this, SPA constructs lexicographically ordered preference pairs and fine-tunes the model using an uncertainty-weighted alignment loss that emphasizes high-confidence, high-gap decisions. Experiments across multiple benchmarks show that SPA improves helpfulness without compromising safety, outperforming strong baselines while preserving general capabilities. Our results demonstrate that SPA provides a scalable and interpretable alignment strategy for critical LLM applications. 

---
# LLMs Do Not See Age: Assessing Demographic Bias in Automated Systematic Review Synthesis 

**Authors**: Favour Yahdii Aghaebe, Tanefa Apekey, Elizabeth Williams, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2511.06000)  

**Abstract**: Clinical interventions often hinge on age: medications and procedures safe for adults may be harmful to children or ineffective for older adults. However, as language models are increasingly integrated into biomedical evidence synthesis workflows, it remains uncertain whether these systems preserve such crucial demographic distinctions. To address this gap, we evaluate how well state-of-the-art language models retain age-related information when generating abstractive summaries of biomedical studies. We construct DemogSummary, a novel age-stratified dataset of systematic review primary studies, covering child, adult, and older adult populations. We evaluate three prominent summarisation-capable LLMs, Qwen (open-source), Longformer (open-source) and GPT-4.1 Nano (proprietary), using both standard metrics and a newly proposed Demographic Salience Score (DSS), which quantifies age-related entity retention and hallucination. Our results reveal systematic disparities across models and age groups: demographic fidelity is lowest for adult-focused summaries, and under-represented populations are more prone to hallucinations. These findings highlight the limitations of current LLMs in faithful and bias-free summarisation and point to the need for fairness-aware evaluation frameworks and summarisation pipelines in biomedical NLP. 

---
# Multi-Reward GRPO Fine-Tuning for De-biasing Large Language Models: A Study Based on Chinese-Context Discrimination Data 

**Authors**: Deng Yixuan, Ji Xiaoqiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06023)  

**Abstract**: Large Language Models (LLMs) often exhibit implicit biases and discriminatory tendencies that reflect underlying social stereotypes. While recent alignment techniques such as RLHF and DPO have mitigated some of these issues, they remain limited in addressing culturally specific and multi-dimensional forms of discrimination. This paper proposes a Multi-Reward Group Relative Policy Optimization (GRPO) framework to fine-tune LLMs toward ethical and bias-free behavior. Our approach constructs a synthetic English-language dataset derived from Chinese-context discrimination categories, including regional, ethnic, and occupational biases. Each instance is paired with both neutral and biased responses to train a reward model based on DeBERTa-v3, which provides multi-dimensional reward signals capturing fairness, neutrality, and linguistic quality. The trained reward model then guides GRPO fine-tuning to optimize model outputs along these ethical dimensions. Experimental results demonstrate significant reductions in bias intensity and improved alignment with non-discriminatory standards without compromising fluency or informativeness. This study highlights the effectiveness of GRPO-based multi-reward optimization for de-biasing LLMs and offers a replicable framework for cultural-contextual ethical alignment. 

---
# MCP4IFC: IFC-Based Building Design Using Large Language Models 

**Authors**: Bharathi Kannan Nithyanantham, Tobias Sesterhenn, Ashwin Nedungadi, Sergio Peral Garijo, Janis Zenkner, Christian Bartelt, Stefan Lüdtke  

**Link**: [PDF](https://arxiv.org/pdf/2511.05533)  

**Abstract**: Bringing generative AI into the architecture, engineering and construction (AEC) field requires systems that can translate natural language instructions into actions on standardized data models. We present MCP4IFC, a comprehensive open-source framework that enables Large Language Models (LLMs) to directly manipulate Industry Foundation Classes (IFC) data through the Model Context Protocol (MCP). The framework provides a set of BIM tools, including scene querying tools for information retrieval, predefined functions for creating and modifying common building elements, and a dynamic code-generation system that combines in-context learning with retrieval-augmented generation (RAG) to handle tasks beyond the predefined toolset. Experiments demonstrate that an LLM using our framework can successfully perform complex tasks, from building a simple house to querying and editing existing IFC data. Our framework is released as open-source to encourage research in LLM-driven BIM design and provide a foundation for AI-assisted modeling workflows. Our code is available at this https URL. 

---
# Multi-Scale Feature Fusion and Graph Neural Network Integration for Text Classification with Large Language Models 

**Authors**: Xiangchen Song, Yulin Huang, Jinxu Guo, Yuchen Liu, Yaxuan Luan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05752)  

**Abstract**: This study investigates a hybrid method for text classification that integrates deep feature extraction from large language models, multi-scale fusion through feature pyramids, and structured modeling with graph neural networks to enhance performance in complex semantic contexts. First, the large language model captures contextual dependencies and deep semantic representations of the input text, providing a rich feature foundation for subsequent modeling. Then, based on multi-level feature representations, the feature pyramid mechanism effectively integrates semantic features of different scales, balancing global information and local details to construct hierarchical semantic expressions. Furthermore, the fused features are transformed into graph representations, and graph neural networks are employed to capture latent semantic relations and logical dependencies in the text, enabling comprehensive modeling of complex interactions among semantic units. On this basis, the readout and classification modules generate the final category predictions. The proposed method demonstrates significant advantages in robustness alignment experiments, outperforming existing models on ACC, F1-Score, AUC, and Precision, which verifies the effectiveness and stability of the framework. This study not only constructs an integrated framework that balances global and local information as well as semantics and structure, but also provides a new perspective for multi-scale feature fusion and structured semantic modeling in text classification tasks. 

---
# CG-TTRL: Context-Guided Test-Time Reinforcement Learning for On-Device Large Language Models 

**Authors**: Peyman Hosseini, Ondrej Bohdal, Taha Ceritli, Ignacio Castro, Matthew Purver, Mete Ozay, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2511.06430)  

**Abstract**: Test-time Reinforcement Learning (TTRL) has shown promise in adapting foundation models for complex tasks at test-time, resulting in large performance improvements. TTRL leverages an elegant two-phase sampling strategy: first, multi-sampling derives a pseudo-label via majority voting, while subsequent downsampling and reward-based fine-tuning encourages the model to explore and learn diverse valid solutions, with the pseudo-label modulating the reward signal. Meanwhile, in-context learning has been widely explored at inference time and demonstrated the ability to enhance model performance without weight updates. However, TTRL's two-phase sampling strategy under-utilizes contextual guidance, which can potentially improve pseudo-label accuracy in the initial exploitation phase while regulating exploration in the second. To address this, we propose context-guided TTRL (CG-TTRL), integrating context dynamically into both sampling phases and propose a method for efficient context selection for on-device applications. Our evaluations on mathematical and scientific QA benchmarks show CG-TTRL outperforms TTRL (e.g. additional 7% relative accuracy improvement over TTRL), while boosting efficiency by obtaining strong performance after only a few steps of test-time training (e.g. 8% relative improvement rather than 1% over TTRL after 3 steps). 

---
# ELEGANCE: Efficient LLM Guidance for Audio-Visual Target Speech Extraction 

**Authors**: Wenxuan Wu, Shuai Wang, Xixin Wu, Helen Meng, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.06288)  

**Abstract**: Audio-visual target speaker extraction (AV-TSE) models primarily rely on visual cues from the target speaker. However, humans also leverage linguistic knowledge, such as syntactic constraints, next word prediction, and prior knowledge of conversation, to extract target speech. Inspired by this observation, we propose ELEGANCE, a novel framework that incorporates linguistic knowledge from large language models (LLMs) into AV-TSE models through three distinct guidance strategies: output linguistic constraints, intermediate linguistic prediction, and input linguistic prior. Comprehensive experiments with RoBERTa, Qwen3-0.6B, and Qwen3-4B on two AV-TSE backbones demon- strate the effectiveness of our approach. Significant improvements are observed in challenging scenarios, including visual cue impaired, unseen languages, target speaker switches, increased interfering speakers, and out-of-domain test set. Demo page: this https URL. 

---
# MCP-RiskCue: Can LLM infer risk information from MCP server System Logs? 

**Authors**: Jiayi Fu, Qiyao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.05867)  

**Abstract**: Large language models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning from Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83% accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at: this https URL 

---
# The Role of High-Performance GPU Resources in Large Language Model Based Radiology Imaging Diagnosis 

**Authors**: Jyun-Ping Kao  

**Link**: [PDF](https://arxiv.org/pdf/2509.16328)  

**Abstract**: Large-language models (LLMs) are rapidly being applied to radiology, enabling automated image interpretation and report generation tasks. Their deployment in clinical practice requires both high diagnostic accuracy and low inference latency, which in turn demands powerful hardware. High-performance graphical processing units (GPUs) provide the necessary compute and memory throughput to run large LLMs on imaging data. We review modern GPU architectures (e.g. NVIDIA A100/H100, AMD Instinct MI250X/MI300) and key performance metrics of floating-point throughput, memory bandwidth, VRAM capacity. We show how these hardware capabilities affect radiology tasks: for example, generating reports or detecting findings on CheXpert and MIMIC-CXR images is computationally intensive and benefits from GPU parallelism and tensor-core acceleration. Empirical studies indicate that using appropriate GPU resources can reduce inference time and improve throughput. We discuss practical challenges including privacy, deployment, cost, power and optimization strategies: mixed-precision, quantization, compression, and multi-GPU scaling. Finally, we anticipate that next-generation features (8-bit tensor cores, enhanced interconnect) will further enable on-premise and federated radiology AI. Advancing GPU infrastructure is essential for safe, efficient LLM-based radiology diagnostics. 

---
