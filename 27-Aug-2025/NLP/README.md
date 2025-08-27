# Generative Interfaces for Language Models 

**Authors**: Jiaqi Chen, Yanzhe Zhang, Yutong Zhang, Yijia Shao, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19227)  

**Abstract**: Large language models (LLMs) are increasingly seen as assistants, copilots, and consultants, capable of supporting a wide range of tasks through natural conversation. However, most systems remain constrained by a linear request-response format that often makes interactions inefficient in multi-turn, information-dense, and exploratory tasks. To address these limitations, we propose Generative Interfaces for Language Models, a paradigm in which LLMs respond to user queries by proactively generating user interfaces (UIs) that enable more adaptive and interactive engagement. Our framework leverages structured interface-specific representations and iterative refinements to translate user queries into task-specific UIs. For systematic evaluation, we introduce a multidimensional assessment framework that compares generative interfaces with traditional chat-based ones across diverse tasks, interaction patterns, and query types, capturing functional, interactive, and emotional aspects of user experience. Results show that generative interfaces consistently outperform conversational ones, with humans preferring them in over 70% of cases. These findings clarify when and why users favor generative interfaces, paving the way for future advancements in human-AI interaction. 

---
# Evaluating the Evaluators: Are readability metrics good measures of readability? 

**Authors**: Isabel Cachola, Daniel Khashabi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2508.19221)  

**Abstract**: Plain Language Summarization (PLS) aims to distill complex documents into accessible summaries for non-expert audiences. In this paper, we conduct a thorough survey of PLS literature, and identify that the current standard practice for readability evaluation is to use traditional readability metrics, such as Flesch-Kincaid Grade Level (FKGL). However, despite proven utility in other fields, these metrics have not been compared to human readability judgments in PLS. We evaluate 8 readability metrics and show that most correlate poorly with human judgments, including the most popular metric, FKGL. We then show that Language Models (LMs) are better judges of readability, with the best-performing model achieving a Pearson correlation of 0.56 with human judgments. Extending our analysis to PLS datasets, which contain summaries aimed at non-expert audiences, we find that LMs better capture deeper measures of readability, such as required background knowledge, and lead to different conclusions than the traditional metrics. Based on these findings, we offer recommendations for best practices in the evaluation of plain language summaries. We release our analysis code and survey data. 

---
# VibeVoice Technical Report 

**Authors**: Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, Shaohan Huang, Yan Xia, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.19205)  

**Abstract**: This report presents VibeVoice, a novel model designed to synthesize long-form speech with multiple speakers by employing next-token diffusion, which is a unified method for modeling continuous data by autoregressively generating latent vectors via diffusion. To enable this, we introduce a novel continuous speech tokenizer that, when compared to the popular Encodec model, improves data compression by 80 times while maintaining comparable performance. The tokenizer effectively preserves audio fidelity while significantly boosting computational efficiency for processing long sequences. Thus, VibeVoice can synthesize long-form speech for up to 90 minutes (in a 64K context window length) with a maximum of 4 speakers, capturing the authentic conversational ``vibe'' and surpassing open-source and proprietary dialogue models. 

---
# Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning 

**Authors**: Alan Li, Yixin Liu, Arpan Sarkar, Doug Downey, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2508.19202)  

**Abstract**: Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge. Finally, we conduct a lightweight analysis, comparing our science-focused data composition with concurrent efforts on long CoT SFT, and release SciLit01, a strong 8B baseline for scientific reasoning. 

---
# Do LVLMs Know What They Know? A Systematic Study of Knowledge Boundary Perception in LVLMs 

**Authors**: Zhikai Ding, Shiyu Ni, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2508.19111)  

**Abstract**: Large vision-language models (LVLMs) demonstrate strong visual question answering (VQA) capabilities but are shown to hallucinate. A reliable model should perceive its knowledge boundaries-knowing what it knows and what it does not. This paper investigates LVLMs' perception of their knowledge boundaries by evaluating three types of confidence signals: probabilistic confidence, answer consistency-based confidence, and verbalized confidence. Experiments on three LVLMs across three VQA datasets show that, although LVLMs possess a reasonable perception level, there is substantial room for improvement. Among the three confidences, probabilistic and consistency-based signals are more reliable indicators, while verbalized confidence often leads to overconfidence. To enhance LVLMs' perception, we adapt several established confidence calibration methods from Large Language Models (LLMs) and propose three effective methods. Additionally, we compare LVLMs with their LLM counterparts, finding that jointly processing visual and textual inputs decreases question-answering performance but reduces confidence, resulting in an improved perception level compared to LLMs. 

---
# Beyond the Black Box: Integrating Lexical and Semantic Methods in Quantitative Discourse Analysis with BERTopic 

**Authors**: Thomas Compton  

**Link**: [PDF](https://arxiv.org/pdf/2508.19099)  

**Abstract**: Quantitative Discourse Analysis has seen growing adoption with the rise of Large Language Models and computational tools. However, reliance on black box software such as MAXQDA and NVivo risks undermining methodological transparency and alignment with research goals. This paper presents a hybrid, transparent framework for QDA that combines lexical and semantic methods to enable triangulation, reproducibility, and interpretability. Drawing from a case study in historical political discourse, we demonstrate how custom Python pipelines using NLTK, spaCy, and Sentence Transformers allow fine-grained control over preprocessing, lemmatisation, and embedding generation. We further detail our iterative BERTopic modelling process, incorporating UMAP dimensionality reduction, HDBSCAN clustering, and c-TF-IDF keyword extraction, optimised through parameter tuning and multiple runs to enhance topic coherence and coverage. By juxtaposing precise lexical searches with context-aware semantic clustering, we argue for a multi-layered approach that mitigates the limitations of either method in isolation. Our workflow underscores the importance of code-level transparency, researcher agency, and methodological triangulation in computational discourse studies. Code and supplementary materials are available via GitHub. 

---
# Retrieval-Augmented Generation for Natural Language Art Provenance Searches in the Getty Provenance Index 

**Authors**: Mathew Henrickson  

**Link**: [PDF](https://arxiv.org/pdf/2508.19093)  

**Abstract**: This research presents a Retrieval-Augmented Generation (RAG) framework for art provenance studies, focusing on the Getty Provenance Index. Provenance research establishes the ownership history of artworks, which is essential for verifying authenticity, supporting restitution and legal claims, and understanding the cultural and historical context of art objects. The process is complicated by fragmented, multilingual archival data that hinders efficient retrieval. Current search portals require precise metadata, limiting exploratory searches. Our method enables natural-language and multilingual searches through semantic retrieval and contextual summarization, reducing dependence on metadata structures. We assess RAG's capability to retrieve and summarize auction records using a 10,000-record sample from the Getty Provenance Index - German Sales. The results show this approach provides a scalable solution for navigating art market archives, offering a practical tool for historians and cultural heritage professionals conducting historically sensitive research. 

---
# It's All About In-Context Learning! Teaching Extremely Low-Resource Languages to LLMs 

**Authors**: Yue Li, Zhixue Zhao, Carolina Scarton  

**Link**: [PDF](https://arxiv.org/pdf/2508.19089)  

**Abstract**: Extremely low-resource languages, especially those written in rare scripts, as shown in Figure 1, remain largely unsupported by large language models (LLMs). This is due in part to compounding factors such as the lack of training data. This paper delivers the first comprehensive analysis of whether LLMs can acquire such languages purely via in-context learning (ICL), with or without auxiliary alignment signals, and how these methods compare to parameter-efficient fine-tuning (PEFT). We systematically evaluate 20 under-represented languages across three state-of-the-art multilingual LLMs. Our findings highlight the limitation of PEFT when both language and its script are extremely under-represented by the LLM. In contrast, zero-shot ICL with language alignment is impressively effective on extremely low-resource languages, while few-shot ICL or PEFT is more beneficial for languages relatively better represented by LLMs. For LLM practitioners working on extremely low-resource languages, we summarise guidelines grounded by our results on adapting LLMs to low-resource languages, e.g., avoiding fine-tuning a multilingual model on languages of unseen scripts. 

---
# "Where does it hurt?" - Dataset and Study on Physician Intent Trajectories in Doctor Patient Dialogues 

**Authors**: Tom Röhr, Soumyadeep Roy, Fares Al Mohamad, Jens-Michalis Papaioannou, Wolfgang Nejdl, Felix Gers, Alexander Löser  

**Link**: [PDF](https://arxiv.org/pdf/2508.19077)  

**Abstract**: In a doctor-patient dialogue, the primary objective of physicians is to diagnose patients and propose a treatment plan. Medical doctors guide these conversations through targeted questioning to efficiently gather the information required to provide the best possible outcomes for patients. To the best of our knowledge, this is the first work that studies physician intent trajectories in doctor-patient dialogues. We use the `Ambient Clinical Intelligence Benchmark' (Aci-bench) dataset for our study. We collaborate with medical professionals to develop a fine-grained taxonomy of physician intents based on the SOAP framework (Subjective, Objective, Assessment, and Plan). We then conduct a large-scale annotation effort to label over 5000 doctor-patient turns with the help of a large number of medical experts recruited using Prolific, a popular crowd-sourcing platform. This large labeled dataset is an important resource contribution that we use for benchmarking the state-of-the-art generative and encoder models for medical intent classification tasks. Our findings show that our models understand the general structure of medical dialogues with high accuracy, but often fail to identify transitions between SOAP categories. We also report for the first time common trajectories in medical dialogue structures that provide valuable insights for designing `differential diagnosis' systems. Finally, we extensively study the impact of intent filtering for medical dialogue summarization and observe a significant boost in performance. We make the codes and data, including annotation guidelines, publicly available at this https URL. 

---
# HiPlan: Hierarchical Planning for LLM-Based Agents with Adaptive Global-Local Guidance 

**Authors**: Ziyue Li, Yuan Chang, Gaihong Yu, Xiaoqiu Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.19076)  

**Abstract**: Large language model (LLM)-based agents have demonstrated remarkable capabilities in decision-making tasks, but struggle significantly with complex, long-horizon planning scenarios. This arises from their lack of macroscopic guidance, causing disorientation and failures in complex tasks, as well as insufficient continuous oversight during execution, rendering them unresponsive to environmental changes and prone to deviations. To tackle these challenges, we introduce HiPlan, a hierarchical planning framework that provides adaptive global-local guidance to boost LLM-based agents'decision-making. HiPlan decomposes complex tasks into milestone action guides for general direction and step-wise hints for detailed actions. During the offline phase, we construct a milestone library from expert demonstrations, enabling structured experience reuse by retrieving semantically similar tasks and milestones. In the execution phase, trajectory segments from past milestones are dynamically adapted to generate step-wise hints that align current observations with the milestone objectives, bridging gaps and correcting deviations. Extensive experiments across two challenging benchmarks demonstrate that HiPlan substantially outperforms strong baselines, and ablation studies validate the complementary benefits of its hierarchical components. 

---
# MovieCORE: COgnitive REasoning in Movies 

**Authors**: Gueter Josmy Faure, Min-Hung Chen, Jia-Fong Yeh, Ying Cheng, Hung-Ting Su, Yung-Hao Tang, Shang-Hong Lai, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19026)  

**Abstract**: This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by up to 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at this https URL. 

---
# Automatic Prompt Optimization with Prompt Distillation 

**Authors**: Viktor N. Zhuravlev, Artur R. Khairullin, Ernest A. Dyagin, Alena N. Sitkina, Nikita I. Kulin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18992)  

**Abstract**: Autoprompting is the process of automatically selecting optimized prompts for language models, which is gaining popularity due to the rapid development of prompt engineering driven by extensive research in the field of large language models (LLMs). This paper presents DistillPrompt -- a novel autoprompting method based on large language models that employs a multi-stage integration of task-specific information into prompts using training data. DistillPrompt utilizes distillation, compression, and aggregation operations to explore the prompt space more thoroughly. The method was tested on different datasets for text classification and generation tasks using the t-lite-instruct-0.1 language model. The results demonstrate a significant average improvement (e.g., 20.12% across the entire dataset compared to Grips) in key metrics over existing methods in the field, establishing DistillPrompt as one of the most effective non-gradient approaches in autoprompting. 

---
# Interpretable by AI Mother Tongue: Native Symbolic Reasoning in Neural Models 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18988)  

**Abstract**: We present a framework where neural models develop an AI Mother Tongue, a native symbolic language that simultaneously supports intuitive reasoning, compositional symbol chains, and inherent interpretability. Unlike post-hoc explanation methods, our approach embeds reasoning directly into the model's representations: symbols capture meaningful semantic patterns, chains trace decision paths, and gated induction mechanisms guide selective focus, yielding transparent yet flexible reasoning. We introduce complementary training objectives to enhance symbol purity and decision sparsity, and employ a sequential specialization strategy to first build broad symbolic competence and then refine intuitive judgments. Experiments on AI tasks demonstrate competitive accuracy alongside verifiable reasoning traces, showing that AI Mother Tongue can serve as a unified mechanism for interpretability, intuition, and symbolic reasoning in neural models. 

---
# Diverse And Private Synthetic Datasets Generation for RAG evaluation: A multi-agent framework 

**Authors**: Ilias Driouich, Hongliu Cao, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2508.18929)  

**Abstract**: Retrieval-augmented generation (RAG) systems improve large language model outputs by incorporating external knowledge, enabling more informed and context-aware responses. However, the effectiveness and trustworthiness of these systems critically depends on how they are evaluated, particularly on whether the evaluation process captures real-world constraints like protecting sensitive information. While current evaluation efforts for RAG systems have primarily focused on the development of performance metrics, far less attention has been given to the design and quality of the underlying evaluation datasets, despite their pivotal role in enabling meaningful, reliable assessments. In this work, we introduce a novel multi-agent framework for generating synthetic QA datasets for RAG evaluation that prioritize semantic diversity and privacy preservation. Our approach involves: (1) a Diversity agent leveraging clustering techniques to maximize topical coverage and semantic variability, (2) a Privacy Agent that detects and mask sensitive information across multiple domains and (3) a QA curation agent that synthesizes private and diverse QA pairs suitable as ground truth for RAG evaluation. Extensive experiments demonstrate that our evaluation sets outperform baseline methods in diversity and achieve robust privacy masking on domain-specific datasets. This work offers a practical and ethically aligned pathway toward safer, more comprehensive RAG system evaluation, laying the foundation for future enhancements aligned with evolving AI regulations and compliance standards. 

---
# Affective Polarization across European Parliaments 

**Authors**: Bojan Evkoski, Igor Mozetič, Nikola Ljubešić, Petra Kralj Novak  

**Link**: [PDF](https://arxiv.org/pdf/2508.18916)  

**Abstract**: Affective polarization, characterized by increased negativity and hostility towards opposing groups, has become a prominent feature of political discourse worldwide. Our study examines the presence of this type of polarization in a selection of European parliaments in a fully automated manner. Utilizing a comprehensive corpus of parliamentary speeches from the parliaments of six European countries, we employ natural language processing techniques to estimate parliamentarian sentiment. By comparing the levels of negativity conveyed in references to individuals from opposing groups versus one's own, we discover patterns of affectively polarized interactions. The findings demonstrate the existence of consistent affective polarization across all six European parliaments. Although activity correlates with negativity, there is no observed difference in affective polarization between less active and more active members of parliament. Finally, we show that reciprocity is a contributing mechanism in affective polarization between parliamentarians across all six parliaments. 

---
# Empowering Computing Education Researchers Through LLM-Assisted Content Analysis 

**Authors**: Laurie Gale, Sebastian Mateos Nicolajsen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18872)  

**Abstract**: Computing education research (CER) is often instigated by practitioners wanting to improve both their own and the wider discipline's teaching practice. However, the latter is often difficult as many researchers lack the colleagues, resources, or capacity to conduct research that is generalisable or rigorous enough to advance the discipline. As a result, research methods that enable sense-making with larger volumes of qualitative data, while not increasing the burden on the researcher, have significant potential within CER.
In this discussion paper, we propose such a method for conducting rigorous analysis on large volumes of textual data, namely a variation of LLM-assisted content analysis (LACA). This method combines content analysis with the use of large language models, empowering researchers to conduct larger-scale research which they would otherwise not be able to perform. Using a computing education dataset, we illustrate how LACA could be applied in a reproducible and rigorous manner. We believe this method has potential in CER, enabling more generalisable findings from a wider range of research. This, together with the development of similar methods, can help to advance both the practice and research quality of the CER discipline. 

---
# ReflectivePrompt: Reflective evolution in autoprompting algorithms 

**Authors**: Viktor N. Zhuravlev, Artur R. Khairullin, Ernest A. Dyagin, Alena N. Sitkina, Nikita I. Kulin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18870)  

**Abstract**: Autoprompting is the process of automatically selecting optimized prompts for language models, which has been gaining popularity with the rapid advancement of prompt engineering, driven by extensive research in the field of large language models (LLMs). This paper presents ReflectivePrompt - a novel autoprompting method based on evolutionary algorithms that employs a reflective evolution approach for more precise and comprehensive search of optimal prompts. ReflectivePrompt utilizes short-term and long-term reflection operations before crossover and elitist mutation to enhance the quality of the modifications they introduce. This method allows for the accumulation of knowledge obtained throughout the evolution process and updates it at each epoch based on the current population. ReflectivePrompt was tested on 33 datasets for classification and text generation tasks using open-access large language models: t-lite-instruct-0.1 and gemma3-27b-it. The method demonstrates, on average, a significant improvement (e.g., 28% on BBH compared to EvoPrompt) in metrics relative to current state-of-the-art approaches, thereby establishing itself as one of the most effective solutions in evolutionary algorithm-based autoprompting. 

---
# ConfTuner: Training Large Language Models to Express Their Confidence Verbally 

**Authors**: Yibo Li, Miao Xiong, Jiaying Wu, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18847)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes domains such as science, law, and healthcare, where accurate expressions of uncertainty are essential for reliability and trust. However, current LLMs are often observed to generate incorrect answers with high confidence, a phenomenon known as "overconfidence". Recent efforts have focused on calibrating LLMs' verbalized confidence: i.e., their expressions of confidence in text form, such as "I am 80% confident that...". Existing approaches either rely on prompt engineering or fine-tuning with heuristically generated uncertainty estimates, both of which have limited effectiveness and generalizability. Motivated by the notion of proper scoring rules for calibration in classical machine learning models, we introduce ConfTuner, a simple and efficient fine-tuning method that introduces minimal overhead and does not require ground-truth confidence scores or proxy confidence estimates. ConfTuner relies on a new loss function, tokenized Brier score, which we theoretically prove to be a proper scoring rule, intuitively meaning that it "correctly incentivizes the model to report its true probability of being correct". ConfTuner improves calibration across diverse reasoning tasks and generalizes to black-box models such as GPT-4o. Our results further show that better-calibrated confidence enables downstream gains in self-correction and model cascade, advancing the development of trustworthy LLM systems. The code is available at this https URL. 

---
# Arrows of Math Reasoning Data Synthesis for Large Language Models: Diversity, Complexity and Correctness 

**Authors**: Sirui Chen, Changxin Tian, Binbin Hu, Kunlong Chen, Ziqi Liu, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.18824)  

**Abstract**: Enhancing the mathematical reasoning of large language models (LLMs) demands high-quality training data, yet conventional methods face critical challenges in scalability, cost, and data reliability. To address these limitations, we propose a novel program-assisted synthesis framework that systematically generates a high-quality mathematical corpus with guaranteed diversity, complexity, and correctness. This framework integrates mathematical knowledge systems and domain-specific tools to create executable programs. These programs are then translated into natural language problem-solution pairs and vetted by a bilateral validation mechanism that verifies solution correctness against program outputs and ensures program-problem consistency. We have generated 12.3 million such problem-solving triples. Experiments demonstrate that models fine-tuned on our data significantly improve their inference capabilities, achieving state-of-the-art performance on several benchmark datasets and showcasing the effectiveness of our synthesis approach. 

---
# LLM-based Contrastive Self-Supervised AMR Learning with Masked Graph Autoencoders for Fake News Detection 

**Authors**: Shubham Gupta, Shraban Kumar Chatterjee, Suman Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18819)  

**Abstract**: The proliferation of misinformation in the digital age has led to significant societal challenges. Existing approaches often struggle with capturing long-range dependencies, complex semantic relations, and the social dynamics influencing news dissemination. Furthermore, these methods require extensive labelled datasets, making their deployment resource-intensive. In this study, we propose a novel self-supervised misinformation detection framework that integrates both complex semantic relations using Abstract Meaning Representation (AMR) and news propagation dynamics. We introduce an LLM-based graph contrastive loss (LGCL) that utilizes negative anchor points generated by a Large Language Model (LLM) to enhance feature separability in a zero-shot manner. To incorporate social context, we employ a multi view graph masked autoencoder, which learns news propagation features from social context graph. By combining these semantic and propagation-based features, our approach effectively differentiates between fake and real news in a self-supervised manner. Extensive experiments demonstrate that our self-supervised framework achieves superior performance compared to other state-of-the-art methodologies, even with limited labelled datasets while improving generalizability. 

---
# LaTeXTrans: Structured LaTeX Translation with Multi-Agent Coordination 

**Authors**: Ziming Zhu, Chenglong Wang, Shunjie Xing, Yifu Huo, Fengning Tian, Quan Du, Di Yang, Chunliang Zhang, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18791)  

**Abstract**: Despite the remarkable progress of modern machine translation (MT) systems on general-domain texts, translating structured LaTeX-formatted documents remains a significant challenge. These documents typically interleave natural language with domain-specific syntax, such as mathematical equations, tables, figures, and cross-references, all of which must be accurately preserved to maintain semantic integrity and compilability. In this paper, we introduce LaTeXTrans, a collaborative multi-agent system designed to address this challenge. LaTeXTrans ensures format preservation, structural fidelity, and terminology consistency through six specialized agents: 1) a Parser that decomposes LaTeX into translation-friendly units via placeholder substitution and syntax filtering; 2) a Translator, Validator, Summarizer, and Terminology Extractor that work collaboratively to ensure context-aware, self-correcting, and terminology-consistent translations; 3) a Generator that reconstructs the translated content into well-structured LaTeX documents. Experimental results demonstrate that LaTeXTrans can outperform mainstream MT systems in both translation accuracy and structural fidelity, offering an effective and practical solution for translating LaTeX-formatted documents. 

---
# Controllable Conversational Theme Detection Track at DSTC 12 

**Authors**: Igor Shalyminov, Hang Su, Jake Vincent, Siffi Singh, Jason Cai, James Gung, Raphael Shu, Saab Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2508.18783)  

**Abstract**: Conversational analytics has been on the forefront of transformation driven by the advances in Speech and Natural Language Processing techniques. Rapid adoption of Large Language Models (LLMs) in the analytics field has taken the problems that can be automated to a new level of complexity and scale. In this paper, we introduce Theme Detection as a critical task in conversational analytics, aimed at automatically identifying and categorizing topics within conversations. This process can significantly reduce the manual effort involved in analyzing expansive dialogs, particularly in domains like customer support or sales. Unlike traditional dialog intent detection, which often relies on a fixed set of intents for downstream system logic, themes are intended as a direct, user-facing summary of the conversation's core inquiry. This distinction allows for greater flexibility in theme surface forms and user-specific customizations. We pose Controllable Conversational Theme Detection problem as a public competition track at Dialog System Technology Challenge (DSTC) 12 -- it is framed as joint clustering and theme labeling of dialog utterances, with the distinctive aspect being controllability of the resulting theme clusters' granularity achieved via the provided user preference data. We give an overview of the problem, the associated dataset and the evaluation metrics, both automatic and human. Finally, we discuss the participant teams' submissions and provide insights from those. The track materials (data and code) are openly available in the GitHub repository. 

---
# Harnessing Rule-Based Reinforcement Learning for Enhanced Grammatical Error Correction 

**Authors**: Yilin Li, Xunjian Yin, Yilin Chen, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18780)  

**Abstract**: Grammatical error correction is a significant task in NLP. Traditional methods based on encoder-decoder models have achieved certain success, but the application of LLMs in this field is still underexplored. Current research predominantly relies on supervised fine-tuning to train LLMs to directly generate the corrected sentence, which limits the model's powerful reasoning ability. To address this limitation, we propose a novel framework based on Rule-Based RL. Through experiments on the Chinese datasets, our Rule-Based RL framework achieves \textbf{state-of-the-art }performance, with a notable increase in \textbf{recall}. This result clearly highlights the advantages of using RL to steer LLMs, offering a more controllable and reliable paradigm for future development in GEC. 

---
# ThinkDial: An Open Recipe for Controlling Reasoning Effort in Large Language Models 

**Authors**: Qianyu He, Siyu Yuan, Xuefeng Li, Mingxuan Wang, Jiangjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18773)  

**Abstract**: Large language models (LLMs) with chain-of-thought reasoning have demonstrated remarkable problem-solving capabilities, but controlling their computational effort remains a significant challenge for practical deployment. Recent proprietary systems like OpenAI's gpt-oss series have introduced discrete operational modes for intuitive reasoning control, but the open-source community has largely failed to achieve such capabilities. In this paper, we introduce ThinkDial, the first open-recipe end-to-end framework that successfully implements gpt-oss-style controllable reasoning through discrete operational modes. Our system enables seamless switching between three distinct reasoning regimes: High mode (full reasoning capability), Medium mode (50 percent token reduction with <10 percent performance degradation), and Low mode (75 percent token reduction with <15 percent performance degradation). We achieve this through an end-to-end training paradigm that integrates budget-mode control throughout the entire pipeline: budget-mode supervised fine-tuning that embeds controllable reasoning capabilities directly into the learning process, and two-phase budget-aware reinforcement learning with adaptive reward shaping. Extensive experiments demonstrate that ThinkDial achieves target compression-performance trade-offs with clear response length reductions while maintaining performance thresholds. The framework also exhibits strong generalization capabilities on out-of-distribution tasks. 

---
# Chronological Passage Assembling in RAG framework for Temporal Question Answering 

**Authors**: Byeongjeong Kim, Jeonghyun Park, Joonho Yang, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18748)  

**Abstract**: Long-context question answering over narrative tasks is challenging because correct answers often hinge on reconstructing a coherent timeline of events while preserving contextual flow in a limited context window. Retrieval-augmented generation (RAG) indexing methods aim to address this challenge by selectively retrieving only necessary document segments. However, narrative texts possess unique characteristics that limit the effectiveness of these existing approaches. Specifically, understanding narrative texts requires more than isolated segments, as the broader context and sequential relationships between segments are crucial for comprehension. To address these limitations, we propose ChronoRAG, a novel RAG framework specialized for narrative texts. This approach focuses on two essential aspects: refining dispersed document information into coherent and structured passages, and preserving narrative flow by explicitly capturing and maintaining the temporal order among retrieved passages. We empirically demonstrate the effectiveness of ChronoRAG through experiments on the NarrativeQA dataset, showing substantial improvements in tasks requiring both factual identification and comprehension of complex sequential relationships, underscoring that reasoning over temporal order is crucial in resolving narrative QA. 

---
# M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations 

**Authors**: Qiao Liang, Ying Shen, Tiantian Chen, Lin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18740)  

**Abstract**: Emotion Cause Triplet Extraction in Multimodal Conversations (MECTEC) has recently gained significant attention in social media analysis, aiming to extract emotion utterances, cause utterances, and emotion categories simultaneously. However, the scarcity of related datasets, with only one published dataset featuring highly uniform dialogue scenarios, hinders model development in this field. To address this, we introduce MECAD, the first multimodal, multi-scenario MECTEC dataset, comprising 989 conversations from 56 TV series spanning a wide range of dialogue contexts. In addition, existing MECTEC methods fail to explicitly model emotional and causal contexts and neglect the fusion of semantic information at different levels, leading to performance degradation. In this paper, we propose M3HG, a novel model that explicitly captures emotional and causal contexts and effectively fuses contextual information at both inter- and intra-utterance levels via a multimodal heterogeneous graph. Extensive experiments demonstrate the effectiveness of M3HG compared with existing state-of-the-art methods. The codes and dataset are available at this https URL. 

---
# Beyond Quality: Unlocking Diversity in Ad Headline Generation with Large Language Models 

**Authors**: Chang Wang, Siyu Yan, Depeng Yuan, Yuqi Chen, Yanhua Huang, Yuanhang Zheng, Shuhao Li, Yinqi Zhang, Kedi Chen, Mingrui Zhu, Ruiwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18739)  

**Abstract**: The generation of ad headlines plays a vital role in modern advertising, where both quality and diversity are essential to engage a broad range of audience segments. Current approaches primarily optimize language models for headline quality or click-through rates (CTR), often overlooking the need for diversity and resulting in homogeneous outputs. To address this limitation, we propose DIVER, a novel framework based on large language models (LLMs) that are jointly optimized for both diversity and quality. We first design a semantic- and stylistic-aware data generation pipeline that automatically produces high-quality training pairs with ad content and multiple diverse headlines. To achieve the goal of generating high-quality and diversified ad headlines within a single forward pass, we propose a multi-stage multi-objective optimization framework with supervised fine-tuning (SFT) and reinforcement learning (RL). Experiments on real-world industrial datasets demonstrate that DIVER effectively balances quality and diversity. Deployed on a large-scale content-sharing platform serving hundreds of millions of users, our framework improves advertiser value (ADVV) and CTR by 4.0% and 1.4%. 

---
# EMMM, Explain Me My Model! Explainable Machine Generated Text Detection in Dialogues 

**Authors**: Angela Yifei Yuan, Haoyi Li, Soyeon Caren Han, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18715)  

**Abstract**: The rapid adoption of large language models (LLMs) in customer service introduces new risks, as malicious actors can exploit them to conduct large-scale user impersonation through machine-generated text (MGT). Current MGT detection methods often struggle in online conversational settings, reducing the reliability and interpretability essential for trustworthy AI deployment. In customer service scenarios where operators are typically non-expert users, explanation become crucial for trustworthy MGT detection. In this paper, we propose EMMM, an explanation-then-detection framework that balances latency, accuracy, and non-expert-oriented interpretability. Experimental results demonstrate that EMMM provides explanations accessible to non-expert users, with 70\% of human evaluators preferring its outputs, while achieving competitive accuracy compared to state-of-the-art models and maintaining low latency, generating outputs within 1 second. Our code and dataset are open-sourced at this https URL. 

---
# Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs 

**Authors**: Duy Le, Kent Ziti, Evan Girard-Sun, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18709)  

**Abstract**: Multilingual riddle generation challenges large language models (LLMs) to balance cultural fluency with creative abstraction. Standard prompting strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized riddles or perform shallow paraphrasing. We introduce Adaptive Originality Filtering (AOF), a prompting framework that filters redundant generations using cosine-based similarity rejection, while enforcing lexical novelty and cross-lingual fidelity. Evaluated across three LLMs and four language pairs, AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915} Distinct-2 in Japanese, signaling improved lexical diversity and reduced redundancy compared to other prompting methods and language pairs. Our findings show that semantic rejection can guide culturally grounded, creative generation without task-specific fine-tuning. 

---
# Attention2Probability: Attention-Driven Terminology Probability Estimation for Robust Speech-to-Text System 

**Authors**: Yanfan Du, Jun Zhang, Bin Wang, Jin Qiu, Lu Huang, Yuan Ge, Xiaoqian Liu, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18701)  

**Abstract**: Recent advances in speech large language models (SLMs) have improved speech recognition and translation in general domains, but accurately generating domain-specific terms or neologisms remains challenging. To address this, we propose Attention2Probability: attention-driven terminology probability estimation for robust speech-to-text system, which is lightweight, flexible, and accurate. Attention2Probability converts cross-attention weights between speech and terminology into presence probabilities, and it further employs curriculum learning to enhance retrieval accuracy. Furthermore, to tackle the lack of data for speech-to-text tasks with terminology intervention, we create and release a new speech dataset with terminology to support future research in this area. Experimental results show that Attention2Probability significantly outperforms the VectorDB method on our test set. Specifically, its maximum recall rates reach 92.57% for Chinese and 86.83% for English. This high recall is achieved with a latency of only 8.71ms per query. Intervening in SLMs' recognition and translation tasks using Attention2Probability-retrieved terms improves terminology accuracy by 6-17%, while revealing that the current utilization of terminology by SLMs has limitations. 

---
# Knowing or Guessing? Robust Medical Visual Question Answering via Joint Consistency and Contrastive Learning 

**Authors**: Songtao Jiang, Yuxi Chen, Sibo Song, Yan Zhang, Yeying Jin, Yang Feng, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18687)  

**Abstract**: In high-stakes medical applications, consistent answering across diverse question phrasings is essential for reliable diagnosis. However, we reveal that current Medical Vision-Language Models (Med-VLMs) exhibit concerning fragility in Medical Visual Question Answering, as their answers fluctuate significantly when faced with semantically equivalent rephrasings of medical questions. We attribute this to two limitations: (1) insufficient alignment of medical concepts, leading to divergent reasoning patterns, and (2) hidden biases in training data that prioritize syntactic shortcuts over semantic understanding. To address these challenges, we construct RoMed, a dataset built upon original VQA datasets containing 144k questions with variations spanning word-level, sentence-level, and semantic-level perturbations. When evaluating state-of-the-art (SOTA) models like LLaVA-Med on RoMed, we observe alarming performance drops (e.g., a 40\% decline in Recall) compared to original VQA benchmarks, exposing critical robustness gaps. To bridge this gap, we propose Consistency and Contrastive Learning (CCL), which integrates two key components: (1) knowledge-anchored consistency learning, aligning Med-VLMs with medical knowledge rather than shallow feature patterns, and (2) bias-aware contrastive learning, mitigating data-specific priors through discriminative representation refinement. CCL achieves SOTA performance on three popular VQA benchmarks and notably improves answer consistency by 50\% on the challenging RoMed test set, demonstrating significantly enhanced robustness. Code will be released. 

---
# Tailored Teaching with Balanced Difficulty: Elevating Reasoning in Multimodal Chain-of-Thought via Prompt Curriculum 

**Authors**: Xinglong Yang, Quan Feng, Zhongying Pan, Xiang Chen, Yu Tian, Wentong Li, Shuofei Qiao, Yuxia Geng, Xingyu Zhao, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18673)  

**Abstract**: The effectiveness of Multimodal Chain-of-Thought (MCoT) prompting is often limited by the use of randomly or manually selected examples. These examples fail to account for both model-specific knowledge distributions and the intrinsic complexity of the tasks, resulting in suboptimal and unstable model performance. To address this, we propose a novel framework inspired by the pedagogical principle of "tailored teaching with balanced difficulty". We reframe prompt selection as a prompt curriculum design problem: constructing a well ordered set of training examples that align with the model's current capabilities. Our approach integrates two complementary signals: (1) model-perceived difficulty, quantified through prediction disagreement in an active learning setup, capturing what the model itself finds challenging; and (2) intrinsic sample complexity, which measures the inherent difficulty of each question-image pair independently of any model. By jointly analyzing these signals, we develop a difficulty-balanced sampling strategy that ensures the selected prompt examples are diverse across both dimensions. Extensive experiments conducted on five challenging benchmarks and multiple popular Multimodal Large Language Models (MLLMs) demonstrate that our method yields substantial and consistent improvements and greatly reduces performance discrepancies caused by random sampling, providing a principled and robust approach for enhancing multimodal reasoning. 

---
# Emotion Omni: Enabling Empathetic Speech Response Generation through Large Language Models 

**Authors**: Haoyu Wang, Guangyan Zhang, Jiale Chen, Jingyu Li, Yuehai Wang, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.18655)  

**Abstract**: With the development of speech large language models (speech LLMs), users can now interact directly with assistants via speech. However, most existing models simply convert the response content into speech without fully understanding the rich emotional and paralinguistic cues embedded in the user's query. In many cases, the same sentence can have different meanings depending on the emotional expression. Furthermore, emotional understanding is essential for improving user experience in human-machine interaction. Currently, most speech LLMs with empathetic capabilities are trained on massive datasets. This approach requires vast amounts of data and significant computational resources. Therefore, a key challenge lies in how to develop a speech LLM capable of generating empathetic responses with limited data and without the need for large-scale training. To address this challenge, we propose Emotion Omni, a novel model architecture designed to understand the emotional content of user speech input and generate empathetic speech responses. Additionally, we developed a data generation pipeline based on an open-source TTS framework to construct a 200k emotional dialogue dataset, which supports the construction of an empathetic speech assistant. The demos are available at this https URL 

---
# Breaking the Trade-Off Between Faithfulness and Expressiveness for Large Language Models 

**Authors**: Chenxu Yang, Qingyi Si, Zheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18651)  

**Abstract**: Grounding responses in external knowledge represents an effective strategy for mitigating hallucinations in Large Language Models (LLMs). However, current LLMs struggle to seamlessly integrate knowledge while simultaneously maintaining faithfulness (or fidelity) and expressiveness, capabilities that humans naturally possess. This limitation results in outputs that either lack support from external knowledge, thereby compromising faithfulness, or appear overly verbose and unnatural, thus sacrificing expressiveness. In this work, to break the trade-off between faithfulness and expressiveness, we propose Collaborative Decoding (CoDe), a novel approach that dynamically integrates output probabilities generated with and without external knowledge. This integration is guided by distribution divergence and model confidence, enabling the selective activation of relevant and reliable expressions from the model's internal parameters. Furthermore, we introduce a knowledge-aware reranking mechanism that prevents over-reliance on prior parametric knowledge while ensuring proper utilization of provided external information. Through comprehensive experiments, our plug-and-play CoDe framework demonstrates superior performance in enhancing faithfulness without compromising expressiveness across diverse LLMs and evaluation metrics, validating both its effectiveness and generalizability. 

---
# Thinking Before You Speak: A Proactive Test-time Scaling Approach 

**Authors**: Cong Li, Wenchang Chai, Hejun Wu, Yan Pan, Pengxu Wei, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18648)  

**Abstract**: Large Language Models (LLMs) often exhibit deficiencies with complex reasoning tasks, such as maths, which we attribute to the discrepancy between human reasoning patterns and those presented in the LLMs' training data. When dealing with complex problems, humans tend to think carefully before expressing solutions. However, they often do not articulate their inner thoughts, including their intentions and chosen methodologies. Consequently, critical insights essential for bridging reasoning steps may be absent in training data collected from human sources. To bridge this gap, we proposes inserting \emph{insight}s between consecutive reasoning steps, which review the status and initiate the next reasoning steps. Unlike prior prompting strategies that rely on a single or a workflow of static prompts to facilitate reasoning, \emph{insight}s are \emph{proactively} generated to guide reasoning processes. We implement our idea as a reasoning framework, named \emph{Thinking Before You Speak} (TBYS), and design a pipeline for automatically collecting and filtering in-context examples for the generation of \emph{insight}s, which alleviates human labeling efforts and fine-tuning overheads. Experiments on challenging mathematical datasets verify the effectiveness of TBYS. Project website: this https URL 

---
# Scaling Laws for Task-Stratified Knowledge in Post-Training Quantized Large Language Models 

**Authors**: Chenxi Zhou, Pengfei Cao, Jiang Li, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18609)  

**Abstract**: Large language models (LLMs) present significant deployment challenges due to their scale, with post-training quantization (PTQ) emerging as a practical compression solution. However, a comprehensive understanding of how PTQ precisely impacts diverse LLM knowledge capabilities remains elusive, and existing scaling laws for quantized models often overlook crucial PTQ-specific parameters and task-specific sensitivities. This paper addresses these gaps by conducting an extensive empirical investigation to establish task-stratified scaling laws. We disentangle LLM knowledge into memorization and utilization capabilities and develop a unified quantitative framework that incorporates model size, effective bit-width, calibration set size, and group size. Our central finding reveals that knowledge memorization exhibits markedly greater sensitivity to variations in effective bit-width, calibration set size, and model size compared to the more robust knowledge utilization. These findings offer a fine-grained understanding of PTQ's impact and provide guidance for developing knowledge-aware quantization strategies that can better preserve targeted cognitive functions. 

---
# A New NMT Model for Translating Clinical Texts from English to Spanish 

**Authors**: Rumeng Li, Xun Wang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18607)  

**Abstract**: Translating electronic health record (EHR) narratives from English to Spanish is a clinically important yet challenging task due to the lack of a parallel-aligned corpus and the abundant unknown words contained. To address such challenges, we propose \textbf{NOOV} (for No OOV), a new neural machine translation (NMT) system that requires little in-domain parallel-aligned corpus for training. NOOV integrates a bilingual lexicon automatically learned from parallel-aligned corpora and a phrase look-up table extracted from a large biomedical knowledge resource, to alleviate both the unknown word problem and the word-repeat challenge in NMT, enhancing better phrase generation of NMT systems. Evaluation shows that NOOV is able to generate better translation of EHR with improvement in both accuracy and fluency. 

---
# What do language models model? Transformers, automata, and the format of thought 

**Authors**: Colin Klein  

**Link**: [PDF](https://arxiv.org/pdf/2508.18598)  

**Abstract**: What do large language models actually model? Do they tell us something about human capacities, or are they models of the corpus we've trained them on? I give a non-deflationary defence of the latter position. Cognitive science tells us that linguistic capabilities in humans rely supralinear formats for computation. The transformer architecture, by contrast, supports at best a linear formats for processing. This argument will rely primarily on certain invariants of the computational architecture of transformers. I then suggest a positive story about what transformers are doing, focusing on Liu et al. (2022)'s intriguing speculations about shortcut automata. I conclude with why I don't think this is a terribly deflationary story. Language is not (just) a means for expressing inner state but also a kind of 'discourse machine' that lets us make new language given appropriate context. We have learned to use this technology in one way; LLMs have also learned to use it too, but via very different means. 

---
# The Mind's Eye: A Multi-Faceted Reward Framework for Guiding Visual Metaphor Generation 

**Authors**: Girish A. Koushik, Fatemeh Nazarieh, Katherine Birch, Shenbin Qian, Diptesh Kanojia  

**Link**: [PDF](https://arxiv.org/pdf/2508.18569)  

**Abstract**: Visual metaphor generation is a challenging task that aims to generate an image given an input text metaphor. Inherently, it needs language understanding to bind a source concept with a target concept, in a way that preserves meaning while ensuring visual coherence. We propose a self-evaluating visual metaphor generation framework that focuses on metaphor alignment. Our self-evaluation approach combines existing metrics with our newly proposed metaphor decomposition score and a meaning alignment (MA) metric. Within this setup, we explore two novel approaches: a training-free pipeline that explicitly decomposes prompts into source-target-meaning (S-T-M) mapping for image synthesis, and a complementary training-based pipeline that improves alignment using our proposed self-evaluation reward schema, without any large-scale retraining. On the held-out test set, the training-free approach surpasses strong closed baselines (GPT-4o, Imagen) on decomposition, CLIP, and MA scores, with the training-based approach close behind. We evaluate our framework output using a user-facing study, and observed that participants preferred GPT-4o overall, while our training-free pipeline led open-source methods and edged Imagen on abstract metaphors. Our analyses show S-T-M prompting helps longer or more abstract metaphors, with closed models excelling on short, concrete cases; we also observe sensitivity to sampler settings. Overall, structured prompting and lightweight RL perform metaphor alignment well under modest compute, and remaining gaps to human preference appear driven by aesthetics and sampling. 

---
# COMET-poly: Machine Translation Metric Grounded in Other Candidates 

**Authors**: Maike Züfle, Vilém Zouhar, Tu Anh Dinh, Felipe Maia Polo, Jan Niehues, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18549)  

**Abstract**: Automated metrics for machine translation attempt to replicate human judgment. Unlike humans, who often assess a translation in the context of multiple alternatives, these metrics typically consider only the source sentence and a single translation. This discrepancy in the evaluation setup may negatively impact the performance of automated metrics. We propose two automated metrics that incorporate additional information beyond the single translation. COMET-polycand uses alternative translations of the same source sentence to compare and contrast with the translation at hand, thereby providing a more informed assessment of its quality. COMET-polyic, inspired by retrieval-based in-context learning, takes in translations of similar source texts along with their human-labeled quality scores to guide the evaluation. We find that including a single additional translation in COMET-polycand improves the segment-level metric performance (0.079 to 0.118 Kendall's tau-b correlation), with further gains when more translations are added. Incorporating retrieved examples in COMET-polyic yields similar improvements (0.079 to 0.116 Kendall's tau-b correlation). We release our models publicly. 

---
# Principled Detection of Hallucinations in Large Language Models via Multiple Testing 

**Authors**: Jiawei Li, Akshayaa Magesh, Venugopal V. Veeravalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.18473)  

**Abstract**: While Large Language Models (LLMs) have emerged as powerful foundational models to solve a variety of tasks, they have also been shown to be prone to hallucinations, i.e., generating responses that sound confident but are actually incorrect or even nonsensical. In this work, we formulate the problem of detecting hallucinations as a hypothesis testing problem and draw parallels to the problem of out-of-distribution detection in machine learning models. We propose a multiple-testing-inspired method to solve the hallucination detection problem, and provide extensive experimental results to validate the robustness of our approach against state-of-the-art methods. 

---
# Integrating gender inclusivity into large language models via instruction tuning 

**Authors**: Alina Wróblewska, Bartosz Żuk  

**Link**: [PDF](https://arxiv.org/pdf/2508.18466)  

**Abstract**: Imagine a language with masculine, feminine, and neuter grammatical genders, yet, due to historical and political conventions, masculine forms are predominantly used to refer to men, women and mixed-gender groups. This is the reality of contemporary Polish. A social consequence of this unfair linguistic system is that large language models (LLMs) trained on Polish texts inherit and reinforce this masculine bias, generating gender-imbalanced outputs. This study addresses this issue by tuning LLMs using the IPIS dataset, a collection of human-crafted gender-inclusive proofreading in Polish and Polish-to-English translation instructions. Grounded in a theoretical linguistic framework, we design a system prompt with explicit gender-inclusive guidelines for Polish. In our experiments, we IPIS-tune multilingual LLMs (Llama-8B, Mistral-7B and Mistral-Nemo) and Polish-specific LLMs (Bielik and PLLuM). Our approach aims to integrate gender inclusivity as an inherent feature of these models, offering a systematic solution to mitigate gender bias in Polish language generation. 

---
# How Reliable are LLMs for Reasoning on the Re-ranking task? 

**Authors**: Nafis Tanveer Islam, Zhiming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18444)  

**Abstract**: With the improving semantic understanding capability of Large Language Models (LLMs), they exhibit a greater awareness and alignment with human values, but this comes at the cost of transparency. Although promising results are achieved via experimental analysis, an in-depth understanding of the LLM's internal workings is unavoidable to comprehend the reasoning behind the re-ranking, which provides end users with an explanation that enables them to make an informed decision. Moreover, in newly developed systems with limited user engagement and insufficient ranking data, accurately re-ranking content remains a significant challenge. While various training methods affect the training of LLMs and generate inference, our analysis has found that some training methods exhibit better explainability than others, implying that an accurate semantic understanding has not been learned through all training methods; instead, abstract knowledge has been gained to optimize evaluation, which raises questions about the true reliability of LLMs. Therefore, in this work, we analyze how different training methods affect the semantic understanding of the re-ranking task in LLMs and investigate whether these models can generate more informed textual reasoning to overcome the challenges of transparency or LLMs and limited training data. To analyze the LLMs for re-ranking tasks, we utilize a relatively small ranking dataset from the environment and the Earth science domain to re-rank retrieved content. Furthermore, we also analyze the explainable information to see if the re-ranking can be reasoned using explainability. 

---
# Can Out-of-Distribution Evaluations Uncover Reliance on Shortcuts? A Case Study in Question Answering 

**Authors**: Michal Štefánik, Timothee Mickus, Marek Kadlčík, Michal Spiegel, Josef Kuchař  

**Link**: [PDF](https://arxiv.org/pdf/2508.18407)  

**Abstract**: A majority of recent work in AI assesses models' generalization capabilities through the lens of performance on out-of-distribution (OOD) datasets. Despite their practicality, such evaluations build upon a strong assumption: that OOD evaluations can capture and reflect upon possible failures in a real-world deployment.
In this work, we challenge this assumption and confront the results obtained from OOD evaluations with a set of specific failure modes documented in existing question-answering (QA) models, referred to as a reliance on spurious features or prediction shortcuts.
We find that different datasets used for OOD evaluations in QA provide an estimate of models' robustness to shortcuts that have a vastly different quality, some largely under-performing even a simple, in-distribution evaluation. We partially attribute this to the observation that spurious shortcuts are shared across ID+OOD datasets, but also find cases where a dataset's quality for training and evaluation is largely disconnected. Our work underlines limitations of commonly-used OOD-based evaluations of generalization, and provides methodology and recommendations for evaluating generalization within and beyond QA more robustly. 

---
# Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning 

**Authors**: Jeong-seok Oh, Jay-yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18395)  

**Abstract**: Probabilistic decoding in Large Language Models (LLMs) often yields inconsistent outputs, particularly on complex or long-form questions. Self-Consistency (SC) mitigates this for short-form QA by majority voting over exact strings, whereas Universal Self-Consistency (USC) and Weighted Unigram Consistency Score (WUCS) extend to long-form responses but lose accuracy on short-form benchmarks.
We introduce Latent Self-Consistency (LSC), which selects the most semantically consistent response using learnable token embeddings. A lightweight forward generation of summary tokens increases inference time by less than 1% and requires no changes to the model architecture.
Across 6 short-form and 5 long-form reasoning benchmarks (e.g., MATH, MMLU, TruthfulQA), LSC surpasses SC, USC and WUCS on all short-form and long-form ones on average, while maintaining negligible computational overhead. These results position LSC as a practical consistency-selection method that works reliably across answer formats. Additionally, LSC provides well-calibrated confidence estimates, maintaining low Expected Calibration Error across both answer formats. 

---
# Integral Transformer: Denoising Attention, Not Too Much Not Too Little 

**Authors**: Ivan Kobyzev, Abbas Ghaddar, Dingtao Hu, Boxing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18387)  

**Abstract**: Softmax self-attention often assigns disproportionate weight to semantically uninformative tokens such as special tokens and punctuation, a phenomenon known as attention noise. While recent methods like Cog Attention and the Differential Transformer have addressed this by introducing negative attention scores, they risk discarding useful information. In this paper, we propose the Integral Transformer, a novel self-attention mechanism that denoises attention by integrating signals sampled from the logit distribution. Our approach mitigates noise while preserving the contributions of special tokens critical for model performance. Extensive experiments demonstrate that our model outperforms vanilla, Cog, and Differential attention variants on well-established knowledge and reasoning language benchmarks. Moreover, our analysis reveals that employing vanilla self-attention in the lower Transformer layers enhances performance and that the Integral Transformer effectively balances attention distributions and reduces rank collapse in upper layers. 

---
# Backprompting: Leveraging Synthetic Production Data for Health Advice Guardrails 

**Authors**: Kellen Tan Cheng, Anna Lisa Gentile, Chad DeLuca, Guang-Jie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.18384)  

**Abstract**: The pervasiveness of large language models (LLMs) in enterprise settings has also brought forth a significant amount of risks associated with their usage. Guardrails technologies aim to mitigate this risk by filtering LLMs' input/output text through various detectors. However, developing and maintaining robust detectors faces many challenges, one of which is the difficulty in acquiring production-quality labeled data on real LLM outputs prior to deployment. In this work, we propose backprompting, a simple yet intuitive solution to generate production-like labeled data for health advice guardrails development. Furthermore, we pair our backprompting method with a sparse human-in-the-loop clustering technique to label the generated data. Our aim is to construct a parallel corpus roughly representative of the original dataset yet resembling real LLM output. We then infuse existing datasets with our synthetic examples to produce robust training data for our detector. We test our technique in one of the most difficult and nuanced guardrails: the identification of health advice in LLM output, and demonstrate improvement versus other solutions. Our detector is able to outperform GPT-4o by up to 3.73%, despite having 400x less parameters. 

---
# Language-Specific Layer Matters: Efficient Multilingual Enhancement for Large Vision-Language Models 

**Authors**: Yuchun Fan, Yilin Wang, Yongyu Mu, Lei Huang, Bei Li, Xiaocheng Feng, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18381)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated exceptional capabilities in understanding visual information with human languages but also exhibit an imbalance in multilingual capabilities. In this work, we delve into the multilingual working pattern of LVLMs and identify a salient correlation between the multilingual understanding ability of LVLMs and language-specific neuron activations in shallow layers. Building on this insight, we introduce PLAST, a training recipe that achieves efficient multilingual enhancement for LVLMs by Precise LAnguage-Specific layers fine-Tuning. PLAST first identifies layers involved in multilingual understanding by monitoring language-specific neuron activations. These layers are then precisely fine-tuned with question-translation pairs to achieve multilingual alignment. Our empirical results on MM-Bench and MMMB demonstrate that PLAST effectively improves the multilingual capabilities of LVLMs and achieves significant efficiency with only 14% of the parameters tuned. Further analysis reveals that PLAST can be generalized to low-resource and complex visual reasoning tasks, facilitating the language-specific visual information engagement in shallow layers. 

---
# Not All Visitors are Bilingual: A Measurement Study of the Multilingual Web from an Accessibility Perspective 

**Authors**: Masudul Hasan Masud Bhuiyan, Matteo Varvello, Yasir Zaki, Cristian-Alexandru Staicu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18328)  

**Abstract**: English is the predominant language on the web, powering nearly half of the world's top ten million websites. Support for multilingual content is nevertheless growing, with many websites increasingly combining English with regional or native languages in both visible content and hidden metadata. This multilingualism introduces significant barriers for users with visual impairments, as assistive technologies like screen readers frequently lack robust support for non-Latin scripts and misrender or mispronounce non-English text, compounding accessibility challenges across diverse linguistic contexts. Yet, large-scale studies of this issue have been limited by the lack of comprehensive datasets on multilingual web content. To address this gap, we introduce LangCrUX, the first large-scale dataset of 120,000 popular websites across 12 languages that primarily use non-Latin scripts. Leveraging this dataset, we conduct a systematic analysis of multilingual web accessibility and uncover widespread neglect of accessibility hints. We find that these hints often fail to reflect the language diversity of visible content, reducing the effectiveness of screen readers and limiting web accessibility. We finally propose Kizuki, a language-aware automated accessibility testing extension to account for the limited utility of language-inconsistent accessibility hints. 

---
# LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions 

**Authors**: Maojia Song, Tej Deep Pala, Weisheng Jin, Amir Zadeh, Chuan Li, Dorien Herremans, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2508.18321)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: this https URL. 

---
# Semantic Attractors and the Emergence of Meaning: Towards a Teleological Model of AGI 

**Authors**: Hans-Joachim Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2508.18290)  

**Abstract**: This essay develops a theoretical framework for a semantic Artificial General Intelligence (AGI) based on the notion of semantic attractors in complex-valued meaning spaces. Departing from current transformer-based language models, which operate on statistical next-token prediction, we explore a model in which meaning is not inferred probabilistically but formed through recursive tensorial transformation. Using cyclic operations involving the imaginary unit \emph{i}, we describe a rotational semantic structure capable of modeling irony, homonymy, and ambiguity. At the center of this model, however, is a semantic attractor -- a teleological operator that, unlike statistical computation, acts as an intentional agent (Microvitum), guiding meaning toward stability, clarity, and expressive depth. Conceived in terms of gradient flows, tensor deformations, and iterative matrix dynamics, the attractor offers a model of semantic transformation that is not only mathematically suggestive, but also philosophically significant. We argue that true meaning emerges not from simulation, but from recursive convergence toward semantic coherence, and that this requires a fundamentally new kind of cognitive architecture -- one designed to shape language, not just predict it. 

---
# StepWiser: Stepwise Generative Judges for Wiser Reasoning 

**Authors**: Wei Xiong, Wenting Zhao, Weizhe Yuan, Olga Golovneva, Tong Zhang, Jason Weston, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2508.19229)  

**Abstract**: As models increasingly leverage multi-step reasoning strategies to solve complex problems, supervising the logical validity of these intermediate steps has become a critical research challenge. Process reward models address this by providing step-by-step feedback, but current approaches have two major drawbacks: they typically function as classifiers without providing explanations, and their reliance on supervised fine-tuning with static datasets limits generalization. Inspired by recent advances, we reframe stepwise reward modeling from a classification task to a reasoning task itself. We thus propose a generative judge that reasons about the policy model's reasoning steps (i.e., meta-reasons), outputting thinking tokens before delivering a final verdict. Our model, StepWiser, is trained by reinforcement learning using relative outcomes of rollouts. We show it provides (i) better judgment accuracy on intermediate steps than existing methods; (ii) can be used to improve the policy model at training time; and (iii) improves inference-time search. 

---
# The Ramon Llull's Thinking Machine for Automated Ideation 

**Authors**: Xinran Zhao, Boyuan Zheng, Chenglei Si, Haofei Yu, Ken Liu, Runlong Zhou, Ruochen Li, Tong Chen, Xiang Li, Yiming Zhang, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19200)  

**Abstract**: This paper revisits Ramon Llull's Ars combinatoria - a medieval framework for generating knowledge through symbolic recombination - as a conceptual foundation for building a modern Llull's thinking machine for research ideation. Our approach defines three compositional axes: Theme (e.g., efficiency, adaptivity), Domain (e.g., question answering, machine translation), and Method (e.g., adversarial training, linear attention). These elements represent high-level abstractions common in scientific work - motivations, problem settings, and technical approaches - and serve as building blocks for LLM-driven exploration. We mine elements from human experts or conference papers and show that prompting LLMs with curated combinations produces research ideas that are diverse, relevant, and grounded in current literature. This modern thinking machine offers a lightweight, interpretable tool for augmenting scientific creativity and suggests a path toward collaborative ideation between humans and AI. 

---
# Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark 

**Authors**: Yuxuan Cai, Yipeng Hao, Jie Zhou, Hang Yan, Zhikai Lei, Rui Zhen, Zhenhua Han, Yutao Yang, Junsong Li, Qianjun Pan, Tianyu Huai, Qin Chen, Xin Li, Kai Chen, Bo Zhang, Xipeng Qiu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2508.19005)  

**Abstract**: As AI advances toward general intelligence, the focus is shifting from systems optimized for static tasks to creating open-ended agents that learn continuously. In this paper, we introduce Experience-driven Lifelong Learning (ELL), a framework for building self-evolving agents capable of continuous growth through real-world interaction. The framework is built on four core principles: (1) Experience Exploration: Agents learn through continuous, self-motivated interaction with dynamic environments, navigating interdependent tasks and generating rich experiential trajectories. (2) Long-term Memory: Agents preserve and structure historical knowledge, including personal experiences, domain expertise, and commonsense reasoning, into a persistent memory system. (3) Skill Learning: Agents autonomously improve by abstracting recurring patterns from experience into reusable skills, which are actively refined and validated for application in new tasks. (4) Knowledge Internalization: Agents internalize explicit and discrete experiences into implicit and intuitive capabilities as "second nature".
We also introduce StuLife, a benchmark dataset for ELL that simulates a student's holistic college journey, from enrollment to academic and personal development, across three core phases and ten detailed sub-scenarios. StuLife is designed around three key paradigm shifts: From Passive to Proactive, From Context to Memory, and From Imitation to Learning. In this dynamic environment, agents must acquire and distill practical skills and maintain persistent memory to make decisions based on evolving state variables. StuLife provides a comprehensive platform for evaluating lifelong learning capabilities, including memory retention, skill transfer, and self-motivated behavior. Beyond evaluating SOTA LLMs on the StuLife benchmark, we also explore the role of context engineering in advancing AGI. 

---
# The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization 

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2508.18976)  

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially. 

---
# Beyond the Textual: Generating Coherent Visual Options for MCQs 

**Authors**: Wanqiang Wang, Longzhu He, Wei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18772)  

**Abstract**: Multiple-choice questions (MCQs) play a crucial role in fostering deep thinking and knowledge integration in education. However, previous research has primarily focused on generating MCQs with textual options, but it largely overlooks the visual options. Moreover, generating high-quality distractors remains a major challenge due to the high cost and limited scalability of manual authoring. To tackle these problems, we propose a Cross-modal Options Synthesis (CmOS), a novel framework for generating educational MCQs with visual options. Our framework integrates Multimodal Chain-of-Thought (MCoT) reasoning process and Retrieval-Augmented Generation (RAG) to produce semantically plausible and visually similar answer and distractors. It also includes a discrimination module to identify content suitable for visual options. Experimental results on test tasks demonstrate the superiority of CmOS in content discrimination, question generation and visual option generation over existing methods across various subjects and educational levels. 

---
# Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models 

**Authors**: Yi Liu, Xiangyu Liu, Zequn Sun, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18760)  

**Abstract**: Large reasoning models (LRMs) have shown remarkable progress on complex reasoning tasks. However, some questions posed to LRMs are inherently unanswerable, such as math problems lacking sufficient conditions. We find that LRMs continually fail to provide appropriate abstentions when confronted with these unanswerable questions. In this paper, we systematically analyze, investigate, and resolve this issue for trustworthy AI. We first conduct a detailed analysis of the distinct response behaviors of LRMs when facing unanswerable questions. Then, we show that LRMs possess sufficient cognitive capabilities to recognize the flaws in these questions. However, they fail to exhibit appropriate abstention behavior, revealing a misalignment between their internal cognition and external response. Finally, to resolve this issue, we propose a lightweight, two-stage method that combines cognitive monitoring with inference-time intervention. Experimental results demonstrate that our method significantly improves the abstention rate while maintaining the overall reasoning performance. 

---
# Text to Query Plans for Question Answering on Large Tables 

**Authors**: Yipeng Zhang, Chen Wang, Yuzhe Zhang, Jacky Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18758)  

**Abstract**: Efficient querying and analysis of large tabular datasets remain significant challenges, especially for users without expertise in programming languages like SQL. Text-to-SQL approaches have shown promising performance on benchmark data; however, they inherit SQL's drawbacks, including inefficiency with large datasets and limited support for complex data analyses beyond basic querying. We propose a novel framework that transforms natural language queries into query plans. Our solution is implemented outside traditional databases, allowing us to support classical SQL commands while avoiding SQL's inherent limitations. Additionally, we enable complex analytical functions, such as principal component analysis and anomaly detection, providing greater flexibility and extensibility than traditional SQL capabilities. We leverage LLMs to iteratively interpret queries and construct operation sequences, addressing computational complexity by incrementally building solutions. By executing operations directly on the data, we overcome context length limitations without requiring the entire dataset to be processed by the model. We validate our framework through experiments on both standard databases and large scientific tables, demonstrating its effectiveness in handling extensive datasets and performing sophisticated data analyses. 

---
# CAC-CoT: Connector-Aware Compact Chain-of-Thought for Efficient Reasoning Data Synthesis Across Dual-System Cognitive Tasks 

**Authors**: Sunguk Choi, Yonghoon Kwon, Heondeuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18743)  

**Abstract**: Long chain-of-thought (CoT) prompting helps Large Language Models (LLMs) solve difficult problems, but very long traces often slow or even degrade performance on fast, intuitive "System-1" tasks. We introduce Connector-Aware Compact CoT (CAC-CoT) -- a method that deliberately restricts reasoning to a small, fixed set of connector phrases, steering the model toward concise and well -- structured explanations. Despite its simplicity, our synthetic method with Gemini-2.0-Flash yields a high-quality training quality. CAC-CoT achieves approximately 85% on GSM8K and approximately 40% on GPQA (System-2) while retaining approximately 90% on S1-Bench (System-1). Its reasoning traces average approximately 300 tokens(ART), about one-third the length of baseline traces, delivering higher efficiency without loss of accuracy. 

---
# Bias Mitigation Agent: Optimizing Source Selection for Fair and Balanced Knowledge Retrieval 

**Authors**: Karanbir Singh, Deepak Muppiri, William Ngu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18724)  

**Abstract**: Large Language Models (LLMs) have transformed the field of artificial intelligence by unlocking the era of generative applications. Built on top of generative AI capabilities, Agentic AI represents a major shift toward autonomous, goal-driven systems that can reason, retrieve, and act. However, they also inherit the bias present in both internal and external information sources. This significantly affects the fairness and balance of retrieved information, and hence reduces user trust. To address this critical challenge, we introduce a novel Bias Mitigation Agent, a multi-agent system designed to orchestrate the workflow of bias mitigation through specialized agents that optimize the selection of sources to ensure that the retrieved content is both highly relevant and minimally biased to promote fair and balanced knowledge dissemination. The experimental results demonstrate an 81.82\% reduction in bias compared to a baseline naive retrieval strategy. 

---
# FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation 

**Authors**: Shaswata Mitra, Azim Bazarov, Martin Duclos, Sudip Mittal, Aritran Piplai, Md Rayhanur Rahman, Edward Zieglar, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18684)  

**Abstract**: Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation. 

---
# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks 

**Authors**: Taishi Nakamura, Satoki Ishikawa, Masaki Kawamura, Takumi Okamoto, Daisuke Nohara, Jun Suzuki, Rio Yokota  

**Link**: [PDF](https://arxiv.org/pdf/2508.18672)  

**Abstract**: Empirical scaling laws have driven the evolution of large language models (LLMs), yet their coefficients shift whenever the model architecture or data pipeline changes. Mixture-of-Experts (MoE) models, now standard in state-of-the-art systems, introduce a new sparsity dimension that current dense-model frontiers overlook. We investigate how MoE sparsity influences two distinct capability regimes: memorization and reasoning. We train families of MoE Transformers that systematically vary total parameters, active parameters, and top-$k$ routing while holding the compute budget fixed. For every model we record pre-training loss, downstream task loss, and task accuracy, allowing us to separate the train-test generalization gap from the loss-accuracy gap. Memorization benchmarks improve monotonically with total parameters, mirroring training loss. By contrast, reasoning performance saturates and can even regress despite continued gains in both total parameters and training loss. Altering top-$k$ alone has little effect when active parameters are constant, and classic hyperparameters such as learning rate and initialization modulate the generalization gap in the same direction as sparsity. Neither post-training reinforcement learning (GRPO) nor extra test-time compute rescues the reasoning deficit of overly sparse models. Our model checkpoints, code and logs are open-source at this https URL. 

---
# Membership Inference Attacks on LLM-based Recommender Systems 

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18665)  

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots. 

---
# UniC-RAG: Universal Knowledge Corruption Attacks to Retrieval-Augmented Generation 

**Authors**: Runpeng Geng, Yanting Wang, Ying Chen, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.18652)  

**Abstract**: Retrieval-augmented generation (RAG) systems are widely deployed in real-world applications in diverse domains such as finance, healthcare, and cybersecurity. However, many studies showed that they are vulnerable to knowledge corruption attacks, where an attacker can inject adversarial texts into the knowledge database of a RAG system to induce the LLM to generate attacker-desired outputs. Existing studies mainly focus on attacking specific queries or queries with similar topics (or keywords). In this work, we propose UniC-RAG, a universal knowledge corruption attack against RAG systems. Unlike prior work, UniC-RAG jointly optimizes a small number of adversarial texts that can simultaneously attack a large number of user queries with diverse topics and domains, enabling an attacker to achieve various malicious objectives, such as directing users to malicious websites, triggering harmful command execution, or launching denial-of-service attacks. We formulate UniC-RAG as an optimization problem and further design an effective solution to solve it, including a balanced similarity-based clustering method to enhance the attack's effectiveness. Our extensive evaluations demonstrate that UniC-RAG is highly effective and significantly outperforms baselines. For instance, UniC-RAG could achieve over 90% attack success rate by injecting 100 adversarial texts into a knowledge database with millions of texts to simultaneously attack a large set of user queries (e.g., 2,000). Additionally, we evaluate existing defenses and show that they are insufficient to defend against UniC-RAG, highlighting the need for new defense mechanisms in RAG systems. 

---
# Beyond Benchmark: LLMs Evaluation with an Anthropomorphic and Value-oriented Roadmap 

**Authors**: Jun Wang, Ninglun Gu, Kailai Zhang, Zijiao Zhang, Yelun Bao, Jin Yang, Xu Yin, Liwei Liu, Yihuan Liu, Pengyong Li, Gary G. Yen, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18646)  

**Abstract**: For Large Language Models (LLMs), a disconnect persists between benchmark performance and real-world utility. Current evaluation frameworks remain fragmented, prioritizing technical metrics while neglecting holistic assessment for deployment. This survey introduces an anthropomorphic evaluation paradigm through the lens of human intelligence, proposing a novel three-dimensional taxonomy: Intelligence Quotient (IQ)-General Intelligence for foundational capacity, Emotional Quotient (EQ)-Alignment Ability for value-based interactions, and Professional Quotient (PQ)-Professional Expertise for specialized proficiency. For practical value, we pioneer a Value-oriented Evaluation (VQ) framework assessing economic viability, social impact, ethical alignment, and environmental sustainability. Our modular architecture integrates six components with an implementation roadmap. Through analysis of 200+ benchmarks, we identify key challenges including dynamic assessment needs and interpretability gaps. It provides actionable guidance for developing LLMs that are technically proficient, contextually relevant, and ethically sound. We maintain a curated repository of open-source evaluation resources at: this https URL. 

---
# RLMR: Reinforcement Learning with Mixed Rewards for Creative Writing 

**Authors**: Jianxing Liao, Tian Zhang, Xiao Feng, Yusong Zhang, Rui Yang, Haorui Wang, Bosi Wen, Ziying Wang, Runzhi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18642)  

**Abstract**: Large language models are extensively utilized in creative writing applications. Creative writing requires a balance between subjective writing quality (e.g., literariness and emotional expression) and objective constraint following (e.g., format requirements and word limits). Existing reinforcement learning methods struggle to balance these two aspects: single reward strategies fail to improve both abilities simultaneously, while fixed-weight mixed-reward methods lack the ability to adapt to different writing scenarios. To address this problem, we propose Reinforcement Learning with Mixed Rewards (RLMR), utilizing a dynamically mixed reward system from a writing reward model evaluating subjective writing quality and a constraint verification model assessing objective constraint following. The constraint following reward weight is adjusted dynamically according to the writing quality within sampled groups, ensuring that samples violating constraints get negative advantage in GRPO and thus penalized during training, which is the key innovation of this proposed method. We conduct automated and manual evaluations across diverse model families from 8B to 72B parameters. Additionally, we construct a real-world writing benchmark named WriteEval for comprehensive evaluation. Results illustrate that our method achieves consistent improvements in both instruction following (IFEval from 83.36\% to 86.65\%) and writing quality (72.75\% win rate in manual expert pairwise evaluations on WriteEval). To the best of our knowledge, RLMR is the first work to combine subjective preferences with objective verification in online RL training, providing an effective solution for multi-dimensional creative writing optimization. 

---
# Designing across domains with declarative thinking: Insights from the 96-Eyes ptychographic imager project 

**Authors**: Antony C Chan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18512)  

**Abstract**: This article presents a practitioner's reflection on applying declarative, 5th generation, problem formulation language (5GL) to de novo imaging system design, informed by experiences across the interdisciplinary research in academia and cross-functional product development within the private sector. Using the 96-Eyes project: 96-camera parallel multi-modal imager for high-throughput drug discovery as a representative case, I illustrate how project requirements, ranging from hardware constraints to life sciences needs, can be formalized into machine-readable problem statements to preserve mission-critical input from diverse domain stakeholders. This declarative approach enhances transparency, ensures design traceability, and minimizes costly misalignment across optical, algorithmic, hardware-accelerated compute, and life sciences teams.
Alongside the technical discussion of 5GL with real-world code examples, I reflect on the practical barriers to adopting 5GL in environments where imperative, 3rd-generation languages (3GL) remain the default medium for inter-team collaboration. Rather than offering an one-size-fits-all solution, these learned lessons highlight how programming paradigms implicitly shapes research workflows through existing domain hierarchies. The discussion aims to invite further explorations into how declarative problem formulations can facilitate innovation in settings where concurrent R\&{}D workflows are gaining traction, as opposed to environments where sequential, phase-driven workflows remain the norm. 

---
# A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs 

**Authors**: Anders Mølmen Høst, Pierre Lison, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18439)  

**Abstract**: Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable.
This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient.
Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping. 

---
# Training Language Model Agents to Find Vulnerabilities with CTF-Dojo 

**Authors**: Terry Yue Zhuo, Dingmin Wang, Hantian Ding, Varun Kumar, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18370)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-Dojo, achieving up to 11.6% absolute gains over strong baselines across three competitive benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best-performing 32B model reaches 31.9% Pass@1, establishing a new open-weight state-of-the-art that rivals frontier models like DeepSeek-V3-0324 and Gemini-2.5-Flash. By framing CTF-style tasks as a benchmark for executable-agent learning, CTF-Dojo demonstrates that execution-grounded training signals are not only effective but pivotal in advancing high-performance ML agents without dependence on costly proprietary systems. 

---
# SALMAN: Stability Analysis of Language Models Through the Maps Between Graph-based Manifolds 

**Authors**: Wuxinlin Cheng, Yupeng Cao, Jinwen Wu, Koduvayur Subbalakshmi, Tian Han, Zhuo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18306)  

**Abstract**: Recent strides in pretrained transformer-based language models have propelled state-of-the-art performance in numerous NLP tasks. Yet, as these models grow in size and deployment, their robustness under input perturbations becomes an increasingly urgent question. Existing robustness methods often diverge between small-parameter and large-scale models (LLMs), and they typically rely on labor-intensive, sample-specific adversarial designs. In this paper, we propose a unified, local (sample-level) robustness framework (SALMAN) that evaluates model stability without modifying internal parameters or resorting to complex perturbation heuristics. Central to our approach is a novel Distance Mapping Distortion (DMD) measure, which ranks each sample's susceptibility by comparing input-to-output distance mappings in a near-linear complexity manner. By demonstrating significant gains in attack efficiency and robust training, we position our framework as a practical, model-agnostic tool for advancing the reliability of transformer-based NLP systems. 

---
# Can VLMs Recall Factual Associations From Visual References? 

**Authors**: Dhananjay Ashok, Ashutosh Chaubey, Hirona J. Arai, Jonathan May, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2508.18297)  

**Abstract**: Through a controlled study, we identify a systematic deficiency in the multimodal grounding of Vision Language Models (VLMs). While VLMs can recall factual associations when provided a textual reference to an entity; their ability to do so is significantly diminished when the reference is visual instead. Forcing VLMs to rely on image representations of an entity halves their ability to recall factual knowledge, suggesting that VLMs struggle to link their internal knowledge of an entity with its image representation. We show that such linking failures are correlated with the expression of distinct patterns in model internal states, and that probes on these internal states achieve over 92% accuracy at flagging cases where the VLM response is unreliable. These probes can be applied, without retraining, to identify when a VLM will fail to correctly answer a question that requires an understanding of multimodal input. When used to facilitate selective prediction on a visual question answering task, the probes increase coverage by 7.87% (absolute) while also reducing the risk of error by 0.9% (absolute). Addressing the systematic, detectable deficiency is an important avenue in language grounding, and we provide informed recommendations for future directions. 

---
# H-PRM: A Pluggable Hotword Pre-Retrieval Module for Various Speech Recognition Systems 

**Authors**: Huangyu Dai, Lingtao Mao, Ben Chen, Zihan Wang, Zihan Liang, Ying Han, Chenyi Lei, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.18295)  

**Abstract**: Hotword customization is crucial in ASR to enhance the accuracy of domain-specific terms. It has been primarily driven by the advancements in traditional models and Audio large language models (LLMs). However, existing models often struggle with large-scale hotwords, as the recognition rate drops dramatically with the number of hotwords increasing. In this paper, we introduce a novel hotword customization system that utilizes a hotword pre-retrieval module (H-PRM) to identify the most relevant hotword candidate by measuring the acoustic similarity between the hotwords and the speech segment. This plug-and-play solution can be easily integrated into traditional models such as SeACo-Paraformer, significantly enhancing hotwords post-recall rate (PRR). Additionally, we incorporate H-PRM into Audio LLMs through a prompt-based approach, enabling seamless customization of hotwords. Extensive testing validates that H-PRM can outperform existing methods, showing a new direction for hotword customization in ASR. 

---
# Toward Responsible ASR for African American English Speakers: A Scoping Review of Bias and Equity in Speech Technology 

**Authors**: Jay L. Cunningham, Adinawa Adjagbodjou, Jeffrey Basoah, Jainaba Jawara, Kowe Kadoma, Aaleyah Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.18288)  

**Abstract**: This scoping literature review examines how fairness, bias, and equity are conceptualized and operationalized in Automatic Speech Recognition (ASR) and adjacent speech and language technologies (SLT) for African American English (AAE) speakers and other linguistically diverse communities. Drawing from 44 peer-reviewed publications across Human-Computer Interaction (HCI), Machine Learning/Natural Language Processing (ML/NLP), and Sociolinguistics, we identify four major areas of inquiry: (1) how researchers understand ASR-related harms; (2) inclusive data practices spanning collection, curation, annotation, and model training; (3) methodological and theoretical approaches to linguistic inclusion; and (4) emerging practices and design recommendations for more equitable systems. While technical fairness interventions are growing, our review highlights a critical gap in governance-centered approaches that foreground community agency, linguistic justice, and participatory accountability. We propose a governance-centered ASR lifecycle as an emergent interdisciplinary framework for responsible ASR development and offer implications for researchers, practitioners, and policymakers seeking to address language marginalization in speech AI systems. 

---
