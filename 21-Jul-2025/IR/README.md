# DUALRec: A Hybrid Sequential and Language Model Framework for Context-Aware Movie Recommendation 

**Authors**: Yitong Li, Raoul Grasman  

**Link**: [PDF](https://arxiv.org/pdf/2507.13957)  

**Abstract**: The modern recommender systems are facing an increasing challenge of modelling and predicting the dynamic and context-rich user preferences. Traditional collaborative filtering and content-based methods often struggle to capture the temporal patternings and evolving user intentions. While Large Language Models (LLMs) have gained gradual attention in recent years, by their strong semantic understanding and reasoning abilities, they are not inherently designed to model chronologically evolving user preference and intentions. On the other hand, for sequential models like LSTM (Long-Short-Term-Memory) which is good at capturing the temporal dynamics of user behaviour and evolving user preference over time, but still lacks a rich semantic understanding for comprehensive recommendation generation. In this study, we propose DUALRec (Dynamic User-Aware Language-based Recommender), a novel recommender that leverages the complementary strength of both models, which combines the temporal modelling abilities of LSTM networks with semantic reasoning power of the fine-tuned Large Language Models. The LSTM component will capture users evolving preference through their viewing history, while the fine-tuned LLM variants will leverage these temporal user insights to generate next movies that users might enjoy. Experimental results on MovieLens-1M dataset shows that the DUALRec model outperforms a wide range of baseline models, with comprehensive evaluation matrices of Hit Rate (HR@k), Normalized Discounted Cumulative Gain (NDCG@k), and genre similarity metrics. This research proposes a novel architecture that bridges the gap between temporal sequence modeling and semantic reasoning, and offers a promising direction for developing more intelligent and context-aware recommenders. 

---
# PARK: Personalized academic retrieval with knowledge-graphs 

**Authors**: Pranav Kasela, Gabriella Pasi, Raffaele Perego  

**Link**: [PDF](https://arxiv.org/pdf/2507.13910)  

**Abstract**: Academic Search is a search task aimed to manage and retrieve scientific documents like journal articles and conference papers. Personalization in this context meets individual researchers' needs by leveraging, through user profiles, the user related information (e.g. documents authored by a researcher), to improve search effectiveness and to reduce the information overload. While citation graphs are a valuable means to support the outcome of recommender systems, their use in personalized academic search (with, e.g. nodes as papers and edges as citations) is still under-explored.
Existing personalized models for academic search often struggle to fully capture users' academic interests. To address this, we propose a two-step approach: first, training a neural language model for retrieval, then converting the academic graph into a knowledge graph and embedding it into a shared semantic space with the language model using translational embedding techniques. This allows user models to capture both explicit relationships and hidden structures in citation graphs and paper content. We evaluate our approach in four academic search domains, outperforming traditional graph-based and personalized models in three out of four, with up to a 10\% improvement in MAP@100 over the second-best model. This highlights the potential of knowledge graph-based user models to enhance retrieval effectiveness. 

---
# SPARQL Query Generation with LLMs: Measuring the Impact of Training Data Memorization and Knowledge Injection 

**Authors**: Aleksandr Gashkov, Aleksandr Perevalov, Maria Eltsova, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2507.13859)  

**Abstract**: Nowadays, the importance of software with natural-language user interfaces cannot be underestimated. In particular, in Question Answering (QA) systems, generating a SPARQL query for a given natural-language question (often named Query Building) from the information retrieved from the same question is the central task of QA systems working over Knowledge Graphs (KGQA). Due to the rise of Large Language Models (LLMs), they are considered a well-suited method to increase the quality of the question-answering functionality, as there is still a lot of room for improvement, aiming for enhanced quality and trustworthiness. However, LLMs are trained on web data, where researchers have no control over whether the benchmark or the knowledge graph was already included in the training data. In this paper, we introduce a novel method that evaluates the quality of LLMs by generating a SPARQL query from a natural-language question under various conditions: (1) zero-shot SPARQL generation, (2) with knowledge injection, and (3) with "anonymized" knowledge injection. This enables us, for the first time, to estimate the influence of the training data on the QA quality improved by LLMs. Ultimately, this will help to identify how portable a method is or whether good results might mostly be achieved because a benchmark was already included in the training data (cf. LLM memorization). The developed method is portable, robust, and supports any knowledge graph; therefore, it could be easily applied to any KGQA or LLM, s.t., generating consistent insights into the actual LLM capabilities is possible. 

---
# RAG-based Architectures for Drug Side Effect Retrieval in LLMs 

**Authors**: Shad Nygren, Pinar Avci, Andre Daniels, Reza Rassol, Afshin Beheshti, Diego Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2507.13822)  

**Abstract**: Drug side effects are a major global health concern, necessitating advanced methods for their accurate detection and analysis. While Large Language Models (LLMs) offer promising conversational interfaces, their inherent limitations, including reliance on black-box training data, susceptibility to hallucinations, and lack of domain-specific knowledge, hinder their reliability in specialized fields like pharmacovigilance. To address this gap, we propose two architectures: Retrieval-Augmented Generation (RAG) and GraphRAG, which integrate comprehensive drug side effect knowledge into a Llama 3 8B language model. Through extensive evaluations on 19,520 drug side effect associations (covering 976 drugs and 3,851 side effect terms), our results demonstrate that GraphRAG achieves near-perfect accuracy in drug side effect retrieval. This framework offers a highly accurate and scalable solution, signifying a significant advancement in leveraging LLMs for critical pharmacovigilance applications. 

---
# Point of Interest Recommendation: Pitfalls and Viable Solutions 

**Authors**: Alejandro Bellogín, Linus W. Dietz, Francesco Ricci, Pablo Sánchez  

**Link**: [PDF](https://arxiv.org/pdf/2507.13725)  

**Abstract**: Point of interest (POI) recommendation can play a pivotal role in enriching tourists' experiences by suggesting context-dependent and preference-matching locations and activities, such as restaurants, landmarks, itineraries, and cultural attractions. Unlike some more common recommendation domains (e.g., music and video), POI recommendation is inherently high-stakes: users invest significant time, money, and effort to search, choose, and consume these suggested POIs. Despite the numerous research works in the area, several fundamental issues remain unresolved, hindering the real-world applicability of the proposed approaches. In this paper, we discuss the current status of the POI recommendation problem and the main challenges we have identified. The first contribution of this paper is a critical assessment of the current state of POI recommendation research and the identification of key shortcomings across three main dimensions: datasets, algorithms, and evaluation methodologies. We highlight persistent issues such as the lack of standardized benchmark datasets, flawed assumptions in the problem definition and model design, and inadequate treatment of biases in the user behavior and system performance. The second contribution is a structured research agenda that, starting from the identified issues, introduces important directions for future work related to multistakeholder design, context awareness, data collection, trustworthiness, novel interactions, and real-world evaluation. 

---
# IP2: Entity-Guided Interest Probing for Personalized News Recommendation 

**Authors**: Youlin Wu, Yuanyuan Sun, Xiaokun Zhang, Haoxi Zhan, Bo Xu, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.13622)  

**Abstract**: News recommender systems aim to provide personalized news reading experiences for users based on their reading history. Behavioral science studies suggest that screen-based news reading contains three successive steps: scanning, title reading, and then clicking. Adhering to these steps, we find that intra-news entity interest dominates the scanning stage, while the inter-news entity interest guides title reading and influences click decisions. Unfortunately, current methods overlook the unique utility of entities in news recommendation. To this end, we propose a novel method called IP2 to probe entity-guided reading interest at both intra- and inter-news levels. At the intra-news level, a Transformer-based entity encoder is devised to aggregate mentioned entities in the news title into one signature entity. Then, a signature entity-title contrastive pre-training is adopted to initialize entities with proper meanings using the news story context, which in the meantime facilitates us to probe for intra-news entity interest. As for the inter-news level, a dual tower user encoder is presented to capture inter-news reading interest from both the title meaning and entity sides. In addition to highlighting the contribution of inter-news entity guidance, a cross-tower attention link is adopted to calibrate title reading interest using inter-news entity interest, thus further aligning with real-world behavior. Extensive experiments on two real-world datasets demonstrate that our IP2 achieves state-of-the-art performance in news recommendation. 

---
# Revisiting Prompt Engineering: A Comprehensive Evaluation for LLM-based Personalized Recommendation 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2507.13525)  

**Abstract**: Large language models (LLMs) can perform recommendation tasks by taking prompts written in natural language as input. Compared to traditional methods such as collaborative filtering, LLM-based recommendation offers advantages in handling cold-start, cross-domain, and zero-shot scenarios, as well as supporting flexible input formats and generating explanations of user behavior. In this paper, we focus on a single-user setting, where no information from other users is used. This setting is practical for privacy-sensitive or data-limited applications. In such cases, prompt engineering becomes especially important for controlling the output generated by the LLM. We conduct a large-scale comparison of 23 prompt types across 8 public datasets and 12 LLMs. We use statistical tests and linear mixed-effects models to evaluate both accuracy and inference cost. Our results show that for cost-efficient LLMs, three types of prompts are especially effective: those that rephrase instructions, consider background knowledge, and make the reasoning process easier to follow. For high-performance LLMs, simple prompts often outperform more complex ones while reducing cost. In contrast, commonly used prompting styles in natural language processing, such as step-by-step reasoning, or the use of reasoning models often lead to lower accuracy. Based on these findings, we provide practical suggestions for selecting prompts and LLMs depending on the required balance between accuracy and cost. 

---
# DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning 

**Authors**: Qingyun Sun, Jiaqi Yuan, Shan He, Xiao Guan, Haonan Yuan, Xingcheng Fu, Jianxin Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13396)  

**Abstract**: Graph Retrieval-Augmented Generation has emerged as a powerful paradigm for grounding large language models with external structured knowledge. However, existing Graph RAG methods struggle with temporal reasoning, due to their inability to model the evolving structure and order of real-world events. In this work, we introduce DyG-RAG, a novel event-centric dynamic graph retrieval-augmented generation framework designed to capture and reason over temporal knowledge embedded in unstructured text. To eliminate temporal ambiguity in traditional retrieval units, DyG-RAG proposes Dynamic Event Units (DEUs) that explicitly encode both semantic content and precise temporal anchors, enabling accurate and interpretable time-aware retrieval. To capture temporal and causal dependencies across events, DyG-RAG constructs an event graph by linking DEUs that share entities and occur close in time, supporting efficient and meaningful multi-hop reasoning. To ensure temporally consistent generation, DyG-RAG introduces an event timeline retrieval pipeline that retrieves event sequences via time-aware traversal, and proposes a Time Chain-of-Thought strategy for temporally grounded answer generation. This unified pipeline enables DyG-RAG to retrieve coherent, temporally ordered event sequences and to answer complex, time-sensitive queries that standard RAG systems cannot resolve. Extensive experiments on temporal QA benchmarks demonstrate that DyG-RAG significantly improves the accuracy and recall of three typical types of temporal reasoning questions, paving the way for more faithful and temporal-aware generation. DyG-RAG is available at this https URL. 

---
# Automated Interpretation of Non-Destructive Evaluation Contour Maps Using Large Language Models for Bridge Condition Assessment 

**Authors**: Viraj Nishesh Darji, Callie C. Liao, Duoduo Liao  

**Link**: [PDF](https://arxiv.org/pdf/2507.14107)  

**Abstract**: Bridge maintenance and safety are essential for transportation authorities, and Non-Destructive Evaluation (NDE) techniques are critical to assessing structural integrity. However, interpreting NDE data can be time-consuming and requires expertise, potentially delaying decision-making. Recent advancements in Large Language Models (LLMs) offer new ways to automate and improve this analysis. This pilot study introduces a holistic assessment of LLM capabilities for interpreting NDE contour maps and demonstrates the effectiveness of LLMs in providing detailed bridge condition analyses. It establishes a framework for integrating LLMs into bridge inspection workflows, indicating that LLM-assisted analysis can enhance efficiency without compromising accuracy. In this study, several LLMs are explored with prompts specifically designed to enhance the quality of image descriptions, which are applied to interpret five different NDE contour maps obtained through technologies for assessing bridge conditions. Each LLM model is evaluated based on its ability to produce detailed descriptions, identify defects, provide actionable recommendations, and demonstrate overall accuracy. The research indicates that four of the nine models provide better image descriptions, effectively covering a wide range of topics related to the bridge's condition. The outputs from these four models are summarized using five different LLMs to form a comprehensive overview of the bridge. Notably, LLMs ChatGPT-4 and Claude 3.5 Sonnet generate more effective summaries. The findings suggest that LLMs have the potential to significantly improve efficiency and accuracy. This pilot study presents an innovative approach that leverages LLMs for image captioning in parallel and summarization, enabling faster decision-making in bridge maintenance and enhancing infrastructure management and safety assessments. 

---
# Lessons from the TREC Plain Language Adaptation of Biomedical Abstracts (PLABA) track 

**Authors**: Brian Ondov, William Xia, Kush Attal, Ishita Unde, Jerry He, Hoa Dang, Ian Soboroff, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2507.14096)  

**Abstract**: Objective: Recent advances in language models have shown potential to adapt professional-facing biomedical literature to plain language, making it accessible to patients and caregivers. However, their unpredictability, combined with the high potential for harm in this domain, means rigorous evaluation is necessary. Our goals with this track were to stimulate research and to provide high-quality evaluation of the most promising systems.
Methods: We hosted the Plain Language Adaptation of Biomedical Abstracts (PLABA) track at the 2023 and 2024 Text Retrieval Conferences. Tasks included complete, sentence-level, rewriting of abstracts (Task 1) as well as identifying and replacing difficult terms (Task 2). For automatic evaluation of Task 1, we developed a four-fold set of professionally-written references. Submissions for both Tasks 1 and 2 were provided extensive manual evaluation from biomedical experts.
Results: Twelve teams spanning twelve countries participated in the track, with models from multilayer perceptrons to large pretrained transformers. In manual judgments of Task 1, top-performing models rivaled human levels of factual accuracy and completeness, but not simplicity or brevity. Automatic, reference-based metrics generally did not correlate well with manual judgments. In Task 2, systems struggled with identifying difficult terms and classifying how to replace them. When generating replacements, however, LLM-based systems did well in manually judged accuracy, completeness, and simplicity, though not in brevity.
Conclusion: The PLABA track showed promise for using Large Language Models to adapt biomedical literature for the general public, while also highlighting their deficiencies and the need for improved automatic benchmarking tools. 

---
# DENSE: Longitudinal Progress Note Generation with Temporal Modeling of Heterogeneous Clinical Notes Across Hospital Visits 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.14079)  

**Abstract**: Progress notes are among the most clinically meaningful artifacts in an Electronic Health Record (EHR), offering temporally grounded insights into a patient's evolving condition, treatments, and care decisions. Despite their importance, they are severely underrepresented in large-scale EHR datasets. For instance, in the widely used Medical Information Mart for Intensive Care III (MIMIC-III) dataset, only about $8.56\%$ of hospital visits include progress notes, leaving gaps in longitudinal patient narratives. In contrast, the dataset contains a diverse array of other note types, each capturing different aspects of care.
We present DENSE (Documenting Evolving Progress Notes from Scattered Evidence), a system designed to align with clinical documentation workflows by simulating how physicians reference past encounters while drafting progress notes. The system introduces a fine-grained note categorization and a temporal alignment mechanism that organizes heterogeneous notes across visits into structured, chronological inputs. At its core, DENSE leverages a clinically informed retrieval strategy to identify temporally and semantically relevant content from both current and prior visits. This retrieved evidence is used to prompt a large language model (LLM) to generate clinically coherent and temporally aware progress notes.
We evaluate DENSE on a curated cohort of patients with multiple visits and complete progress note documentation. The generated notes demonstrate strong longitudinal fidelity, achieving a temporal alignment ratio of $1.089$, surpassing the continuity observed in original notes. By restoring narrative coherence across fragmented documentation, our system supports improved downstream tasks such as summarization, predictive modeling, and clinical decision support, offering a scalable solution for LLM-driven note synthesis in real-world healthcare settings. 

---
# Preprint: Did I Just Browse A Website Written by LLMs? 

**Authors**: Sichang "Steven" He, Ramesh Govindan, Harsha V. Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2507.13933)  

**Abstract**: Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are insufficient, because they perform well mainly on clean, prose-like text, while web content has complex markup and diverse genres.
We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem. 

---
# Question-Answer Extraction from Scientific Articles Using Knowledge Graphs and Large Language Models 

**Authors**: Hosein Azarbonyad, Zi Long Zhu, Georgios Cheirmpos, Zubair Afzal, Vikrant Yadav, Georgios Tsatsaronis  

**Link**: [PDF](https://arxiv.org/pdf/2507.13827)  

**Abstract**: When deciding to read an article or incorporate it into their research, scholars often seek to quickly identify and understand its main ideas. In this paper, we aim to extract these key concepts and contributions from scientific articles in the form of Question and Answer (QA) pairs. We propose two distinct approaches for generating QAs. The first approach involves selecting salient paragraphs, using a Large Language Model (LLM) to generate questions, ranking these questions by the likelihood of obtaining meaningful answers, and subsequently generating answers. This method relies exclusively on the content of the articles. However, assessing an article's novelty typically requires comparison with the existing literature. Therefore, our second approach leverages a Knowledge Graph (KG) for QA generation. We construct a KG by fine-tuning an Entity Relationship (ER) extraction model on scientific articles and using it to build the graph. We then employ a salient triplet extraction method to select the most pertinent ERs per article, utilizing metrics such as the centrality of entities based on a triplet TF-IDF-like measure. This measure assesses the saliency of a triplet based on its importance within the article compared to its prevalence in the literature. For evaluation, we generate QAs using both approaches and have them assessed by Subject Matter Experts (SMEs) through a set of predefined metrics to evaluate the quality of both questions and answers. Our evaluations demonstrate that the KG-based approach effectively captures the main ideas discussed in the articles. Furthermore, our findings indicate that fine-tuning the ER extraction model on our scientific corpus is crucial for extracting high-quality triplets from such documents. 

---
# Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations 

**Authors**: Cedric Waterschoot, Nava Tintarev, Francesco Barile  

**Link**: [PDF](https://arxiv.org/pdf/2507.13705)  

**Abstract**: Large Language Models (LLMs) are increasingly being implemented as joint decision-makers and explanation generators for Group Recommender Systems (GRS). In this paper, we evaluate these recommendations and explanations by comparing them to social choice-based aggregation strategies. Our results indicate that LLM-generated recommendations often resembled those produced by Additive Utilitarian (ADD) aggregation. However, the explanations typically referred to averaging ratings (resembling but not identical to ADD aggregation). Group structure, uniform or divergent, did not impact the recommendations. Furthermore, LLMs regularly claimed additional criteria such as user or item similarity, diversity, or used undefined popularity metrics or thresholds. Our findings have important implications for LLMs in the GRS pipeline as well as standard aggregation strategies. Additional criteria in explanations were dependent on the number of ratings in the group scenario, indicating potential inefficiency of standard aggregation methods at larger item set sizes. Additionally, inconsistent and ambiguous explanations undermine transparency and explainability, which are key motivations behind the use of LLMs for GRS. 

---
# Off-Policy Evaluation and Learning for Matching Markets 

**Authors**: Yudai Hayashi, Shuhei Goda, Yuta Saito  

**Link**: [PDF](https://arxiv.org/pdf/2507.13608)  

**Abstract**: Matching users based on mutual preferences is a fundamental aspect of services driven by reciprocal recommendations, such as job search and dating applications. Although A/B tests remain the gold standard for evaluating new policies in recommender systems for matching markets, it is costly and impractical for frequent policy updates. Off-Policy Evaluation (OPE) thus plays a crucial role by enabling the evaluation of recommendation policies using only offline logged data naturally collected on the platform. However, unlike conventional recommendation settings, the large scale and bidirectional nature of user interactions in matching platforms introduce variance issues and exacerbate reward sparsity, making standard OPE methods unreliable. To address these challenges and facilitate effective offline evaluation, we propose novel OPE estimators, \textit{DiPS} and \textit{DPR}, specifically designed for matching markets. Our methods combine elements of the Direct Method (DM), Inverse Propensity Score (IPS), and Doubly Robust (DR) estimators while incorporating intermediate labels, such as initial engagement signals, to achieve better bias-variance control in matching markets. Theoretically, we derive the bias and variance of the proposed estimators and demonstrate their advantages over conventional methods. Furthermore, we show that these estimators can be seamlessly extended to offline policy learning methods for improving recommendation policies for making more matches. We empirically evaluate our methods through experiments on both synthetic data and A/B testing logs from a real job-matching platform. The empirical results highlight the superiority of our approach over existing methods in off-policy evaluation and learning tasks for a variety of configurations. 

---
# Smart Routing for Multimodal Video Retrieval: When to Search What 

**Authors**: Kevin Dela Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2507.13374)  

**Abstract**: We introduce ModaRoute, an LLM-based intelligent routing system that dynamically selects optimal modalities for multimodal video retrieval. While dense text captions can achieve 75.9% Recall@5, they require expensive offline processing and miss critical visual information present in 34% of clips with scene text not captured by ASR. By analyzing query intent and predicting information needs, ModaRoute reduces computational overhead by 41% while achieving 60.9% Recall@5. Our approach uses GPT-4.1 to route queries across ASR (speech), OCR (text), and visual indices, averaging 1.78 modalities per query versus exhaustive 3.0 modality search. Evaluation on 1.8M video clips demonstrates that intelligent routing provides a practical solution for scaling multimodal retrieval systems, reducing infrastructure costs while maintaining competitive effectiveness for real-world deployment. 

---
