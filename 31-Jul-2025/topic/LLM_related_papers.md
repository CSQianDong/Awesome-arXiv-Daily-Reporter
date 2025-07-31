# RecGPT Technical Report 

**Authors**: Chao Yi, Dian Chen, Gaoyang Guo, Jiakai Tang, Jian Wu, Jing Yu, Sunhao Dai, Wen Chen, Wenjun Yang, Yuning Jiang, Zhujin Gao, Bo Zheng, Chi Li, Dimin Wang, Dixuan Wang, Fan Li, Fan Zhang, Haibin Chen, Haozhuang Liu, Jialin Zhu, Jiamang Wang, Jiawei Wu, Jin Cui, Ju Huang, Kai Zhang, Kan Liu, Lang Tian, Liang Rao, Longbin Li, Lulu Zhao, Mao Zhang, Na He, Peiyang Wang, Qiqi Huang, Tao Luo, Wenbo Su, Xiaoxiao He, Xin Tong, Xu Chen, Xunke Xi, Yang Li, Yaxuan Wu, Yeqiu Yang, Yi Hu, Yinnan Song, Yuchen Li, Yujie Luo, Yujin Yuan, Yuliang Yan, Zhengyang Wang, Zhibo Xiao, Zhixin Ma, Zile Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22879)  

**Abstract**: Recommender systems are among the most impactful applications of artificial intelligence, serving as critical infrastructure connecting users, merchants, and platforms. However, most current industrial systems remain heavily reliant on historical co-occurrence patterns and log-fitting objectives, i.e., optimizing for past user interactions without explicitly modeling user intent. This log-fitting approach often leads to overfitting to narrow historical preferences, failing to capture users' evolving and latent interests. As a result, it reinforces filter bubbles and long-tail phenomena, ultimately harming user experience and threatening the sustainability of the whole recommendation ecosystem.
To address these challenges, we rethink the overall design paradigm of recommender systems and propose RecGPT, a next-generation framework that places user intent at the center of the recommendation pipeline. By integrating large language models (LLMs) into key stages of user interest mining, item retrieval, and explanation generation, RecGPT transforms log-fitting recommendation into an intent-centric process. To effectively align general-purpose LLMs to the above domain-specific recommendation tasks at scale, RecGPT incorporates a multi-stage training paradigm, which integrates reasoning-enhanced pre-alignment and self-training evolution, guided by a Human-LLM cooperative judge system. Currently, RecGPT has been fully deployed on the Taobao App. Online experiments demonstrate that RecGPT achieves consistent performance gains across stakeholders: users benefit from increased content diversity and satisfaction, merchants and the platform gain greater exposure and conversions. These comprehensive improvement results across all stakeholders validates that LLM-driven, intent-centric design can foster a more sustainable and mutually beneficial recommendation ecosystem. 

---
# Investigating Hallucination in Conversations for Low Resource Languages 

**Authors**: Amit Das, Md. Najib Hasan, Souvika Sarkar, Zheng Zhang, Fatemeh Jamshidi, Tathagata Bhattacharya, Nilanjana Raychawdhury, Dongji Feng, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2507.22720)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in generating text that closely resemble human writing. However, they often generate factually incorrect statements, a problem typically referred to as 'hallucination'. Addressing hallucination is crucial for enhancing the reliability and effectiveness of LLMs. While much research has focused on hallucinations in English, our study extends this investigation to conversational data in three languages: Hindi, Farsi, and Mandarin. We offer a comprehensive analysis of a dataset to examine both factual and linguistic errors in these languages for GPT-3.5, GPT-4o, Llama-3.1, Gemma-2.0, DeepSeek-R1 and Qwen-3. We found that LLMs produce very few hallucinated responses in Mandarin but generate a significantly higher number of hallucinations in Hindi and Farsi. 

---
# Resource-Efficient Adaptation of Large Language Models for Text Embeddings via Prompt Engineering and Contrastive Fine-tuning 

**Authors**: Benedikt Roth, Stephan Rappensperger, Tianming Qiu, Hamza Imamović, Julian Wörmann, Hao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22729)  

**Abstract**: Large Language Models (LLMs) have become a cornerstone in Natural Language Processing (NLP), achieving impressive performance in text generation. Their token-level representations capture rich, human-aligned semantics. However, pooling these vectors into a text embedding discards crucial information. Nevertheless, many non-generative downstream tasks, such as clustering, classification, or retrieval, still depend on accurate and controllable sentence- or document-level embeddings. We explore several adaptation strategies for pre-trained, decoder-only LLMs: (i) various aggregation techniques for token embeddings, (ii) task-specific prompt engineering, and (iii) text-level augmentation via contrastive fine-tuning. Combining these components yields state-of-the-art performance on the English clustering track of the Massive Text Embedding Benchmark (MTEB). An analysis of the attention map further shows that fine-tuning shifts focus from prompt tokens to semantically relevant words, indicating more effective compression of meaning into the final hidden state. Our experiments demonstrate that LLMs can be effectively adapted as text embedding models through a combination of prompt engineering and resource-efficient contrastive fine-tuning on synthetically generated positive pairs. 

---
# Multilingual Political Views of Large Language Models: Identification and Steering 

**Authors**: Daniil Gurgurov, Katharina Trinley, Ivan Vykopal, Josef van Genabith, Simon Ostermann, Roberto Zamparelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.22623)  

**Abstract**: Large language models (LLMs) are increasingly used in everyday tools and applications, raising concerns about their potential influence on political views. While prior research has shown that LLMs often exhibit measurable political biases--frequently skewing toward liberal or progressive positions--key gaps remain. Most existing studies evaluate only a narrow set of models and languages, leaving open questions about the generalizability of political biases across architectures, scales, and multilingual settings. Moreover, few works examine whether these biases can be actively controlled.
In this work, we address these gaps through a large-scale study of political orientation in modern open-source instruction-tuned LLMs. We evaluate seven models, including LLaMA-3.1, Qwen-3, and Aya-Expanse, across 14 languages using the Political Compass Test with 11 semantically equivalent paraphrases per statement to ensure robust measurement. Our results reveal that larger models consistently shift toward libertarian-left positions, with significant variations across languages and model families. To test the manipulability of political stances, we utilize a simple center-of-mass activation intervention technique and show that it reliably steers model responses toward alternative ideological positions across multiple languages. Our code is publicly available at this https URL. 

---
# Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs 

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22564)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems. 

---
# Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation 

**Authors**: Daniil Gurgurov, Katharina Trinley, Yusser Al Ghussin, Tanja Baeumel, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.22608)  

**Abstract**: Large language models (LLMs) exhibit strong multilingual abilities, yet the neural mechanisms behind language-specific processing remain unclear. We analyze language-specific neurons in Llama-3.1-8B, Mistral-Nemo-12B, and Aya-Expanse-8B & 32B across 21 typologically diverse languages, identifying neurons that control language behavior. Using the Language Activation Probability Entropy (LAPE) method, we show that these neurons cluster in deeper layers, with non-Latin scripts showing greater specialization. Related languages share overlapping neurons, reflecting internal representations of linguistic proximity.
Through language arithmetics, i.e. systematic activation addition and multiplication, we steer models to deactivate unwanted languages and activate desired ones, outperforming simpler replacement approaches. These interventions effectively guide behavior across five multilingual tasks: language forcing, translation, QA, comprehension, and NLI. Manipulation is more successful for high-resource languages, while typological similarity improves effectiveness. We also demonstrate that cross-lingual neuron steering enhances downstream performance and reveal internal "fallback" mechanisms for language selection when neurons are progressively deactivated. Our code is made publicly available at this https URL. 

---
# MASCA: LLM based-Multi Agents System for Credit Assessment 

**Authors**: Gautam Jajoo, Pranjal A Chitale, Saksham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.22758)  

**Abstract**: Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring. 

---
# Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning 

**Authors**: Kwesi Cobbina, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22887)  

**Abstract**: In-context learning (ICL) is a critical emerging capability of large language models (LLMs), enabling few-shot learning during inference by including a few demonstrations (demos) in the prompt. However, it has been found that ICL's performance can be sensitive to the choices of demos and their order. This paper investigates an unexplored new positional bias of ICL for the first time: we observe that the predictions and accuracy can drift drastically when the positions of demos, the system prompt, and the user message in LLM input are varied. We refer to this bias as DEMOS' POSITION IN PROMPT (DPP) bias. We design a systematic evaluation pipeline to study this type of positional bias across classification, question answering, summarization, and reasoning tasks. We introduce two metrics, ACCURACY-CHANGE and PREDICTION-CHANGE, to quantify net gains and output volatility induced by changes in the demos' position. Extensive experiments on ten LLMs from four open-source model families (QWEN, LLAMA3, MISTRAL, COHERE) verify that the bias significantly affects their accuracy and predictions: placing demos at the start of the prompt yields the most stable and accurate outputs with gains of up to +6 points. In contrast, placing demos at the end of the user message flips over 30\% of predictions without improving correctness on QA tasks. Smaller models are most affected by this sensitivity, though even large models remain marginally affected on complex tasks. 

---
# CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset 

**Authors**: Jindřich Libovický, Jindřich Helcl, Andrei Manea, Gianluca Vico  

**Link**: [PDF](https://arxiv.org/pdf/2507.22752)  

**Abstract**: We introduce a benchmark for open-ended regional question answering that encompasses both textual and visual modalities. We also provide strong baselines using state-of-the-art large language models (LLMs). Our dataset consists of manually curated questions and answers grounded in Wikipedia, created by native speakers from Czechia, Slovakia, and Ukraine, with accompanying English translations. It includes both purely textual questions and those requiring visual understanding. As a baseline, we evaluate state-of-the-art LLMs through prompting and complement this with human judgments of answer correctness. Using these human evaluations, we analyze the reliability of existing automatic evaluation metrics. Our baseline results highlight a significant gap in regional knowledge among current LLMs. Moreover, apart from LLM-based evaluation, there is minimal correlation between automated metrics and human judgment. We release this dataset as a resource to (1) assess regional knowledge in LLMs, (2) study cross-lingual generation consistency in a challenging setting, and (3) advance the development of evaluation metrics for open-ended question answering. 

---
# ControlMed: Adding Reasoning Control to Medical Language Model 

**Authors**: Sung-Min Lee, Siyoon Lee, Juyeon Kim, Kyungmin Roh  

**Link**: [PDF](https://arxiv.org/pdf/2507.22545)  

**Abstract**: Reasoning Large Language Models (LLMs) with enhanced accuracy and explainability are increasingly being adopted in the medical domain, as the life-critical nature of clinical decision-making demands reliable support. Despite these advancements, existing reasoning LLMs often generate unnecessarily lengthy reasoning processes, leading to significant computational overhead and response latency. These limitations hinder their practical deployment in real-world clinical environments. To address these challenges, we introduce \textbf{ControlMed}, a medical language model that enables users to actively control the length of the reasoning process at inference time through fine-grained control markers. ControlMed is trained through a three-stage pipeline: 1) pre-training on a large-scale synthetic medical instruction dataset covering both \textit{direct} and \textit{reasoning responses}; 2) supervised fine-tuning with multi-length reasoning data and explicit length-control markers; and 3) reinforcement learning with model-based reward signals to enhance factual accuracy and response quality. Experimental results on a variety of English and Korean medical benchmarks demonstrate that our model achieves similar or better performance compared to state-of-the-art models. Furthermore, users can flexibly balance reasoning accuracy and computational efficiency by controlling the reasoning length as needed. These findings demonstrate that ControlMed is a practical and adaptable solution for clinical question answering and medical information analysis. 

---
# Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance 

**Authors**: Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, Slim Frikha  

**Link**: [PDF](https://arxiv.org/pdf/2507.22448)  

**Abstract**: In this report, we introduce Falcon-H1, a new series of large language models (LLMs) featuring hybrid architecture designs optimized for both high performance and efficiency across diverse use cases. Unlike earlier Falcon models built solely on Transformer or Mamba architectures, Falcon-H1 adopts a parallel hybrid approach that combines Transformer-based attention with State Space Models (SSMs), known for superior long-context memory and computational efficiency. We systematically revisited model design, data strategy, and training dynamics, challenging conventional practices in the field. Falcon-H1 is released in multiple configurations, including base and instruction-tuned variants at 0.5B, 1.5B, 1.5B-deep, 3B, 7B, and 34B parameters. Quantized instruction-tuned models are also available, totaling over 30 checkpoints on Hugging Face Hub. Falcon-H1 models demonstrate state-of-the-art performance and exceptional parameter and training efficiency. The flagship Falcon-H1-34B matches or outperforms models up to 70B scale, such as Qwen3-32B, Qwen2.5-72B, and Llama3.3-70B, while using fewer parameters and less data. Smaller models show similar trends: the Falcon-H1-1.5B-Deep rivals current leading 7B-10B models, and Falcon-H1-0.5B performs comparably to typical 7B models from 2024. These models excel across reasoning, mathematics, multilingual tasks, instruction following, and scientific knowledge. With support for up to 256K context tokens and 18 languages, Falcon-H1 is suitable for a wide range of applications. All models are released under a permissive open-source license, underscoring our commitment to accessible and impactful AI research. 

---
# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records 

**Authors**: Dongchen Li, Jitao Liang, Wei Li, Xiaoyu Wang, Longbing Cao, Kun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22533)  

**Abstract**: Large Language Models (LLMs) hold significant promise for improving clinical decision support and reducing physician burnout by synthesizing complex, longitudinal cancer Electronic Health Records (EHRs). However, their implementation in this critical field faces three primary challenges: the inability to effectively process the extensive length and multilingual nature of patient records for accurate temporal analysis; a heightened risk of clinical hallucination, as conventional grounding techniques such as Retrieval-Augmented Generation (RAG) do not adequately incorporate process-oriented clinical guidelines; and unreliable evaluation metrics that hinder the validation of AI systems in oncology. To address these issues, we propose CliCARE, a framework for Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. The framework operates by transforming unstructured, longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range dependencies, and then grounding the decision support process by aligning these real-world patient trajectories with a normative guideline knowledge graph. This approach provides oncologists with evidence-grounded decision support by generating a high-fidelity clinical summary and an actionable recommendation. We validated our framework using large-scale, longitudinal data from a private Chinese cancer dataset and the public English MIMIC-IV dataset. In these diverse settings, CliCARE significantly outperforms strong baselines, including leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of our results is supported by a robust evaluation protocol, which demonstrates a high correlation with assessments made by expert oncologists. 

---
# PATENTWRITER: A Benchmarking Study for Patent Drafting with LLMs 

**Authors**: Homaira Huda Shomee, Suman Kalyan Maity, Sourav Medya  

**Link**: [PDF](https://arxiv.org/pdf/2507.22387)  

**Abstract**: Large language models (LLMs) have emerged as transformative approaches in several important fields. This paper aims for a paradigm shift for patent writing by leveraging LLMs to overcome the tedious patent-filing process. In this work, we present PATENTWRITER, the first unified benchmarking framework for evaluating LLMs in patent abstract generation. Given the first claim of a patent, we evaluate six leading LLMs -- including GPT-4 and LLaMA-3 -- under a consistent setup spanning zero-shot, few-shot, and chain-of-thought prompting strategies to generate the abstract of the patent. Our benchmark PATENTWRITER goes beyond surface-level evaluation: we systematically assess the output quality using a comprehensive suite of metrics -- standard NLP measures (e.g., BLEU, ROUGE, BERTScore), robustness under three types of input perturbations, and applicability in two downstream patent classification and retrieval tasks. We also conduct stylistic analysis to assess length, readability, and tone. Experimental results show that modern LLMs can generate high-fidelity and stylistically appropriate patent abstracts, often surpassing domain-specific baselines. Our code and dataset are open-sourced to support reproducibility and future research. 

---
# Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs 

**Authors**: Supantho Rakshit, Adele Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.22286)  

**Abstract**: The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs. 

---
# NeedleChain: Measuring Intact Long-Context Reasoning Capability of Large Language Models 

**Authors**: Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22411)  

**Abstract**: The Needle-in-a-Haystack (NIAH) benchmark is widely used to evaluate Large Language Models' (LLMs) ability to understand long contexts (LC). It evaluates the capability to identify query-relevant context within extensive query-irrelevant passages. Although this method serves as a widely accepted standard for evaluating long-context understanding, our findings suggest it may overestimate the true LC capability of LLMs. We demonstrate that even state-of-the-art models such as GPT-4o struggle to intactly incorporate given contexts made up of solely query-relevant ten sentences. In response, we introduce a novel benchmark, \textbf{NeedleChain}, where the context consists entirely of query-relevant information, requiring the LLM to fully grasp the input to answer correctly. Our benchmark allows for flexible context length and reasoning order, offering a more comprehensive analysis of LLM performance. Additionally, we propose an extremely simple yet compelling strategy to improve LC understanding capability of LLM: ROPE Contraction. Our experiments with various advanced LLMs reveal a notable disparity between their ability to process large contexts and their capacity to fully understand them. Source code and datasets are available at this https URL 

---
# Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles 

**Authors**: Kimberly Le Truong, Riccardo Fogliato, Hoda Heidari, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22168)  

**Abstract**: Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations. 

---
# Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations 

**Authors**: Galo Castillo-López, Gaël de Chalendar, Nasredine Semmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.22289)  

**Abstract**: Intent recognition is a fundamental component in task-oriented dialogue systems (TODS). Determining user intents and detecting whether an intent is Out-of-Scope (OOS) is crucial for TODS to provide reliable responses. However, traditional TODS require large amount of annotated data. In this work we propose a hybrid approach to combine BERT and LLMs in zero and few-shot settings to recognize intents and detect OOS utterances. Our approach leverages LLMs generalization power and BERT's computational efficiency in such scenarios. We evaluate our method on multi-party conversation corpora and observe that sharing information from BERT outputs to LLMs leads to system performance improvement. 

---
# A Scalable Pipeline for Estimating Verb Frame Frequencies Using Large Language Models 

**Authors**: Adam M. Morgan, Adeen Flinker  

**Link**: [PDF](https://arxiv.org/pdf/2507.22187)  

**Abstract**: We present an automated pipeline for estimating Verb Frame Frequencies (VFFs), the frequency with which a verb appears in particular syntactic frames. VFFs provide a powerful window into syntax in both human and machine language systems, but existing tools for calculating them are limited in scale, accuracy, or accessibility. We use large language models (LLMs) to generate a corpus of sentences containing 476 English verbs. Next, by instructing an LLM to behave like an expert linguist, we had it analyze the syntactic structure of the sentences in this corpus. This pipeline outperforms two widely used syntactic parsers across multiple evaluation datasets. Furthermore, it requires far fewer resources than manual parsing (the gold-standard), thereby enabling rapid, scalable VFF estimation. Using the LLM parser, we produce a new VFF database with broader verb coverage, finer-grained syntactic distinctions, and explicit estimates of the relative frequencies of structural alternates commonly studied in psycholinguistics. The pipeline is easily customizable and extensible to new verbs, syntactic frames, and even other languages. We present this work as a proof of concept for automated frame frequency estimation, and release all code and data to support future research. 

---
# Traits Run Deep: Enhancing Personality Assessment via Psychology-Guided LLM Representations and Multimodal Apparent Behaviors 

**Authors**: Jia Li, Yichao He, Jiacheng Xu, Tianhao Luo, Zhenzhen Hu, Richang Hong, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22367)  

**Abstract**: Accurate and reliable personality assessment plays a vital role in many fields, such as emotional intelligence, mental health diagnostics, and personalized education. Unlike fleeting emotions, personality traits are stable, often subconsciously leaked through language, facial expressions, and body behaviors, with asynchronous patterns across modalities. It was hard to model personality semantics with traditional superficial features and seemed impossible to achieve effective cross-modal understanding. To address these challenges, we propose a novel personality assessment framework called \textit{\textbf{Traits Run Deep}}. It employs \textit{\textbf{psychology-informed prompts}} to elicit high-level personality-relevant semantic representations. Besides, it devises a \textit{\textbf{Text-Centric Trait Fusion Network}} that anchors rich text semantics to align and integrate asynchronous signals from other modalities. To be specific, such fusion module includes a Chunk-Wise Projector to decrease dimensionality, a Cross-Modal Connector and a Text Feature Enhancer for effective modality fusion and an ensemble regression head to improve generalization in data-scarce situations. To our knowledge, we are the first to apply personality-specific prompts to guide large language models (LLMs) in extracting personality-aware semantics for improved representation quality. Furthermore, extracting and fusing audio-visual apparent behavior features further improves the accuracy. Experimental results on the AVI validation set have demonstrated the effectiveness of the proposed components, i.e., approximately a 45\% reduction in mean squared error (MSE). Final evaluations on the test set of the AVI Challenge 2025 confirm our method's superiority, ranking first in the Personality Assessment track. The source code will be made available at this https URL. 

---
# LLM-Crowdsourced: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models 

**Authors**: Qianhong Guo, Wei Xie, Xiaofang Cai, Enze Wang, Shuoyoucheng Ma, Kai Chen, Xiaofeng Wang, Baosheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22359)  

**Abstract**: Although large language models (LLMs) demonstrate remarkable capabilities across various tasks, evaluating their capabilities remains a challenging task. Existing evaluation methods suffer from issues such as data contamination, black-box operation, and subjective preference. These issues make it difficult to evaluate the LLMs' true capabilities comprehensively. To tackle these challenges, we propose a novel benchmark-free evaluation paradigm, LLM-Crowdsourced. It utilizes LLMs to generate questions, answer independently, and evaluate mutually. This method integrates four key evaluation criteria: dynamic, transparent, objective, and professional, which existing evaluation methods cannot satisfy simultaneously. Experiments on eight mainstream LLMs across mathematics and programming verify the advantages of our method in distinguishing LLM performance. Furthermore, our study reveals several novel findings that are difficult for traditional methods to detect, including but not limited to: (1) Gemini demonstrates the highest original and professional question-design capabilities among others; (2) Some LLMs exhibit ''memorization-based answering'' by misrecognizing questions as familiar ones with a similar structure; (3) LLM evaluation results demonstrate high consistency (robustness). 

---
# VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning 

**Authors**: Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22607)  

**Abstract**: Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach. 

---
# Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22565)  

**Abstract**: The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks. 

---
# CIMR: Contextualized Iterative Multimodal Reasoning for Robust Instruction Following in LVLMs 

**Authors**: Yangshu Yuan, Heng Chen, Xinyi Jiang, Christian Ng, Kexin Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22074)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) has enhanced our ability to process and generate human language and visual information. However, these models often struggle with complex, multi-step multi-modal instructions that require logical reasoning, dynamic feedback integration, and iterative self-correction. To address this, we propose CIMR: Contextualized Iterative Multimodal Reasoning, a novel framework that introduces a context-aware iterative reasoning and self-correction module. CIMR operates in two stages: initial reasoning and response generation, followed by iterative refinement using parsed multi-modal feedback. A dynamic fusion module deeply integrates textual, visual, and contextual features at each step. We fine-tune LLaVA-1.5-7B on the Visual Instruction Tuning (VIT) dataset and evaluate CIMR on the newly introduced Multi-modal Action Planning (MAP) dataset. CIMR achieves 91.5% accuracy, outperforming state-of-the-art models such as GPT-4V (89.2%), LLaVA-1.5 (78.5%), MiniGPT-4 (75.3%), and InstructBLIP (72.8%), demonstrating the efficacy of its iterative reasoning and self-correction capabilities in complex tasks. 

---
# CodeEvo: Interaction-Driven Synthesis of Code-centric Data through Hybrid and Iterative Feedback 

**Authors**: Qiushi Sun, Jinyang Gong, Lei Li, Qipeng Guo, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22080)  

**Abstract**: Acquiring high-quality instruction-code pairs is essential for training Large Language Models (LLMs) for code generation. Manually curated data is expensive and inherently limited in scale, motivating the development of code-centric synthesis methods. Yet, current approaches either focus on augmenting existing code or rely on predefined heuristics, both lacking rigorous data validation, which results in synthetic data that is ungrounded, repetitive, or overly simplistic. Inspired by collaborative programming practices, we propose CodeEvo, a framework that synthesizes code data through iterative interactions between two LLM agents: a Coder, which generates candidate code and test cases based on given instructions, and a Reviewer, which guides the synthesis process by producing new instructions and feedback. We further introduce a hybrid feedback mechanism that combines compiler determinism with the generative flexibility of agents, enabling automatic quality control throughout synthesis. Extensive experiments demonstrate that models fine-tuned on CodeEvo data significantly outperform established baselines across code generation benchmarks with various difficulties. In-depth analyses further provide insights from multiple perspectives into effective code-centric data synthesis. 

---
# Prompt Optimization and Evaluation for LLM Automated Red Teaming 

**Authors**: Michael Freenor, Lauren Alvarez, Milton Leal, Lily Smith, Joel Garrett, Yelyzaveta Husieva, Madeline Woodruff, Ryan Miller, Erich Kummerfeld, Rafael Medeiros, Sander Schulhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.22133)  

**Abstract**: Applications that use Large Language Models (LLMs) are becoming widespread, making the identification of system vulnerabilities increasingly important. Automated Red Teaming accelerates this effort by using an LLM to generate and execute attacks against target systems. Attack generators are evaluated using the Attack Success Rate (ASR) the sample mean calculated over the judgment of success for each attack. In this paper, we introduce a method for optimizing attack generator prompts that applies ASR to individual attacks. By repeating each attack multiple times against a randomly seeded target, we measure an attack's discoverability the expectation of the individual attack success. This approach reveals exploitable patterns that inform prompt optimization, ultimately enabling more robust evaluation and refinement of generators. 

---
# Strategic Deflection: Defending LLMs from Logit Manipulation 

**Authors**: Yassine Rachidy, Jihad Rbaiti, Youssef Hmamouche, Faissal Sehbaoui, Amal El Fallah Seghrouchni  

**Link**: [PDF](https://arxiv.org/pdf/2507.22160)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats. 

---
# Opportunities and Challenges of LLMs in Education: An NLP Perspective 

**Authors**: Sowmya Vajjala, Bashar Alhafni, Stefano Bannò, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.22753)  

**Abstract**: Interest in the role of large language models (LLMs) in education is increasing, considering the new opportunities they offer for teaching, learning, and assessment. In this paper, we examine the impact of LLMs on educational NLP in the context of two main application scenarios: {\em assistance} and {\em assessment}, grounding them along the four dimensions -- reading, writing, speaking, and tutoring. We then present the new directions enabled by LLMs, and the key challenges to address. We envision that this holistic overview would be useful for NLP researchers and practitioners interested in exploring the role of LLMs in developing language-focused and NLP-enabled educational applications of the future. 

---
# DBLPLink 2.0 -- An Entity Linker for the DBLP Scholarly Knowledge Graph 

**Authors**: Debayan Banerjee, Tilahun Abedissa Taffa, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2507.22811)  

**Abstract**: In this work we present an entity linker for DBLP's 2025 version of RDF-based Knowledge Graph. Compared to the 2022 version, DBLP now considers publication venues as a new entity type called dblp:Stream. In the earlier version of DBLPLink, we trained KG-embeddings and re-rankers on a dataset to produce entity linkings. In contrast, in this work, we develop a zero-shot entity linker using LLMs using a novel method, where we re-rank candidate entities based on the log-probabilities of the "yes" token output at the penultimate layer of the LLM. 

---
# Automatically discovering heuristics in a complex SAT solver with large language models 

**Authors**: Yiwen Sun, Furong Ye, Zhihan Chen, Ke Wei, Shaowei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.22876)  

**Abstract**: Satisfiability problem (SAT) is a cornerstone of computational complexity with broad industrial applications, and it remains challenging to optimize modern SAT solvers in real-world settings due to their intricate architectures. While automatic configuration frameworks have been developed, they rely on manually constrained search spaces and yield limited performance gains. This work introduces a novel paradigm which effectively optimizes complex SAT solvers via Large Language Models (LLMs), and a tool called AutoModSAT is developed. Three fundamental challenges are addressed in order to achieve superior performance: (1) LLM-friendly solver: Systematic guidelines are proposed for developing a modularized solver to meet LLMs' compatibility, emphasizing code simplification, information share and bug reduction; (2) Automatic prompt optimization: An unsupervised automatic prompt optimization method is introduced to advance the diversity of LLMs' output; (3) Efficient search strategy: We design a presearch strategy and an EA evolutionary algorithm for the final efficient and effective discovery of heuristics. Extensive experiments across a wide range of datasets demonstrate that AutoModSAT achieves 50% performance improvement over the baseline solver and achieves 30% superiority against the state-of-the-art (SOTA) solvers. Moreover, AutoModSAT attains a 20% speedup on average compared to parameter-tuned alternatives of the SOTA solvers, showcasing the enhanced capability in handling complex problem instances. This work bridges the gap between AI-driven heuristics discovery and mission-critical system optimization, and provides both methodological advancements and empirically validated results for next-generation complex solver development. 

---
# Enhancing Manufacturing Knowledge Access with LLMs and Context-aware Prompting 

**Authors**: Sebastian Monka, Irlan Grangel-González, Stefan Schmid, Lavdim Halilaj, Marc Rickart, Oliver Rudolph, Rui Dias  

**Link**: [PDF](https://arxiv.org/pdf/2507.22619)  

**Abstract**: Knowledge graphs (KGs) have transformed data management within the manufacturing industry, offering effective means for integrating disparate data sources through shared and structured conceptual schemas. However, harnessing the power of KGs can be daunting for non-experts, as it often requires formulating complex SPARQL queries to retrieve specific information. With the advent of Large Language Models (LLMs), there is a growing potential to automatically translate natural language queries into the SPARQL format, thus bridging the gap between user-friendly interfaces and the sophisticated architecture of KGs. The challenge remains in adequately informing LLMs about the relevant context and structure of domain-specific KGs, e.g., in manufacturing, to improve the accuracy of generated queries. In this paper, we evaluate multiple strategies that use LLMs as mediators to facilitate information retrieval from KGs. We focus on the manufacturing domain, particularly on the Bosch Line Information System KG and the I40 Core Information Model. In our evaluation, we compare various approaches for feeding relevant context from the KG to the LLM and analyze their proficiency in transforming real-world questions into SPARQL queries. Our findings show that LLMs can significantly improve their performance on generating correct and complete queries when provided only the adequate context of the KG schema. Such context-aware prompting techniques help LLMs to focus on the relevant parts of the ontology and reduce the risk of hallucination. We anticipate that the proposed techniques help LLMs to democratize access to complex data repositories and empower informed decision-making in manufacturing settings. 

---
# When Truthful Representations Flip Under Deceptive Instructions? 

**Authors**: Xianxuan Long, Yao Fu, Runchao Li, Mu Sheng, Haotian Yu, Xiaotian Han, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22149)  

**Abstract**: Large language models (LLMs) tend to follow maliciously crafted instructions to generate deceptive responses, posing safety challenges. How deceptive instructions alter the internal representations of LLM compared to truthful ones remains poorly understood beyond output analysis. To bridge this gap, we investigate when and how these representations ``flip'', such as from truthful to deceptive, under deceptive versus truthful/neutral instructions. Analyzing the internal representations of Llama-3.1-8B-Instruct and Gemma-2-9B-Instruct on a factual verification task, we find the model's instructed True/False output is predictable via linear probes across all conditions based on the internal representation. Further, we use Sparse Autoencoders (SAEs) to show that the Deceptive instructions induce significant representational shifts compared to Truthful/Neutral representations (which are similar), concentrated in early-to-mid layers and detectable even on complex datasets. We also identify specific SAE features highly sensitive to deceptive instruction and use targeted visualizations to confirm distinct truthful/deceptive representational subspaces. % Our analysis pinpoints layer-wise and feature-level correlates of instructed dishonesty, offering insights for LLM detection and control. Our findings expose feature- and layer-level signatures of deception, offering new insights for detecting and mitigating instructed dishonesty in LLMs. 

---
# Repair-R1: Better Test Before Repair 

**Authors**: Haichuan Hu, Xiaochen Xie, Quanjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22853)  

**Abstract**: APR (Automated Program Repair) aims to automatically locate program defects, generate patches and validate the repairs. Existing techniques for APR are often combined with LLMs (Large Language Models), which leverages the code-related knowledge of LLMs to improve repair effectiveness. Current LLM-based APR methods typically utilize test cases only during the inference stage, adopting an iterative approach that performs repair first and validates it through test execution afterward. This conventional paradigm neglects two important aspects: the potential contribution of test cases in the training phase, and the possibility of leveraging testing prior to repair. To address this, we propose Repair-R1, which introduces test cases into the model's training phase and shifts test generation to precede repair. The model is required to first generate discriminative test cases that can distinguish defective behaviors, and then perform repair based on these tests. This enables the model to better locate defects and understand the underlying causes of defects, thereby improving repair effectiveness. We implement Repair-R1 with three different backbone models, using RL (reinforcement learning) to co-optimize test generation and bug repair. Experimental results on four widely adopted benchmarks demonstrate the superiority of Repair-R1. Specially, compared to vanilla models, Repair-R1 improves repair success rate by 2.68\% to 48.29\%, test generation success rate by 16.38\% to 53.28\%, and test coverage by 0.78\% to 53.96\%. We publish the code and weights at this https URL and this https URL. 

---
# OFCnetLLM: Large Language Model for Network Monitoring and Alertness 

**Authors**: Hong-Jun Yoon, Mariam Kiran, Danial Ebling, Joe Breen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22711)  

**Abstract**: The rapid evolution of network infrastructure is bringing new challenges and opportunities for efficient network management, optimization, and security. With very large monitoring databases becoming expensive to explore, the use of AI and Generative AI can help reduce costs of managing these datasets. This paper explores the use of Large Language Models (LLMs) to revolutionize network monitoring management by addressing the limitations of query finding and pattern analysis. We leverage LLMs to enhance anomaly detection, automate root-cause analysis, and automate incident analysis to build a well-monitored network management team using AI. Through a real-world example of developing our own OFCNetLLM, based on the open-source LLM model, we demonstrate practical applications of OFCnetLLM in the OFC conference network. Our model is developed as a multi-agent approach and is still evolving, and we present early results here. 

---
# RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment 

**Authors**: Marcos Fuster-Pena, David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2507.22580)  

**Abstract**: Automated Program Repair (APR) seeks to automatically correct software bugs without requiring human intervention. However, existing tools tend to generate patches that satisfy test cases without fixing the underlying bug, those are known as overfitting patches. To address this issue, Automated Patch Correctness Assessment (APCA) attempts to identify overfitting patches generated by APR tools. It can be solved as a static approach, meaning that no additional information is needed beyond the original and fixed code snippets. Current static techniques often struggle with reliability, flexibility and transparency. To address these issues, we introduce RePaCA, a novel static APCA technique that leverages Large Language Models (LLMs) specialized in thinking tasks. Our model is prompted with both buggy and fixed code snippets and guided to generate a Chain of Thought that analyses code differences, reasons about how the patch addresses the root cause, and ultimately provides a binary classification: correct or overfitting. To enhance these reasoning capabilities for the APCA task specifically, the LLM is finetuned using Reinforcement Learning with the Group Relative Policy Optimization algorithm. When evaluated on a standard Defects4J-derived test, our approach achieves state-of-the-art performance, with 83.1% accuracy and an 84.8% F1-score. Furthermore, our model demonstrates superior generalization capabilities when trained on different datasets, outperforming the leading technique. This reasoning capability also provides enhanced explainability for the patch assessment. These findings underscore the considerable promise of finetuned, reasoning LLMs to advance static APCA by enhancing accuracy, generalization, and explainability. 

---
# aLLoyM: A large language model for alloy phase diagram prediction 

**Authors**: Yuna Oikawa, Guillaume Deffrennes, Taichi Abe, Ryo Tamura, Koji Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2507.22558)  

**Abstract**: Large Language Models (LLMs) are general-purpose tools with wide-ranging applications, including in materials science. In this work, we introduce aLLoyM, a fine-tuned LLM specifically trained on alloy compositions, temperatures, and their corresponding phase information. To develop aLLoyM, we curated question-and-answer (Q&A) pairs for binary and ternary phase diagrams using the open-source Computational Phase Diagram Database (CPDDB) and assessments based on CALPHAD (CALculation of PHAse Diagrams). We fine-tuned Mistral, an open-source pre-trained LLM, for two distinct Q&A formats: multiple-choice and short-answer. Benchmark evaluations demonstrate that fine-tuning substantially enhances performance on multiple-choice phase diagram questions. Moreover, the short-answer model of aLLoyM exhibits the ability to generate novel phase diagrams from its components alone, underscoring its potential to accelerate the discovery of previously unexplored materials systems. To promote further research and adoption, we have publicly released the short-answer fine-tuned version of aLLoyM, along with the complete benchmarking Q&A dataset, on Hugging Face. 

---
# What is an "Abstract Reasoner"? Revisiting Experiments and Arguments about Large Language Models 

**Authors**: Tian Yun, Chen Sun, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.22457)  

**Abstract**: Recent work has argued that large language models (LLMs) are not "abstract reasoners", citing their poor zero-shot performance on a variety of challenging tasks as evidence. We revisit these experiments in order to add nuance to the claim. First, we show that while LLMs indeed perform poorly in a zero-shot setting, even tuning a small subset of parameters for input encoding can enable near-perfect performance. However, we also show that this finetuning does not necessarily transfer across datasets. We take this collection of empirical results as an invitation to (re-)open the discussion of what it means to be an "abstract reasoner", and why it matters whether LLMs fit the bill. 

---
# Towards Simulating Social Influence Dynamics with LLM-based Multi-agents 

**Authors**: Hsien-Tsung Lin, Pei-Cing Huang, Chan-Tung Ku, Chan Hsu, Pei-Xuan Shieh, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22467)  

**Abstract**: Recent advancements in Large Language Models offer promising capabilities to simulate complex human social interactions. We investigate whether LLM-based multi-agent simulations can reproduce core human social dynamics observed in online forums. We evaluate conformity dynamics, group polarization, and fragmentation across different model scales and reasoning capabilities using a structured simulation framework. Our findings indicate that smaller models exhibit higher conformity rates, whereas models optimized for reasoning are more resistant to social influence. 

---
# Systematic Evaluation of Knowledge Graph Repair with Large Language Models 

**Authors**: Tung-Wei Lin, Gabe Fierro, Han Li, Tianzhen Hong, Pierluigi Nuzzo, Alberto Sangiovanni-Vinentelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.22419)  

**Abstract**: We present a systematic approach for evaluating the quality of knowledge graph repairs with respect to constraint violations defined in shapes constraint language (SHACL). Current evaluation methods rely on \emph{ad hoc} datasets, which limits the rigorous analysis of repair systems in more general settings. Our method addresses this gap by systematically generating violations using a novel mechanism, termed violation-inducing operations (VIOs). We use the proposed evaluation framework to assess a range of repair systems which we build using large language models. We analyze the performance of these systems across different prompting strategies. Results indicate that concise prompts containing both the relevant violated SHACL constraints and key contextual information from the knowledge graph yield the best performance. 

---
# Towards Interpretable Renal Health Decline Forecasting via Multi-LMM Collaborative Reasoning Framework 

**Authors**: Peng-Yi Wu, Pei-Cing Huang, Ting-Yu Chen, Chantung Ku, Ming-Yen Lin, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22464)  

**Abstract**: Accurate and interpretable prediction of estimated glomerular filtration rate (eGFR) is essential for managing chronic kidney disease (CKD) and supporting clinical decisions. Recent advances in Large Multimodal Models (LMMs) have shown strong potential in clinical prediction tasks due to their ability to process visual and textual information. However, challenges related to deployment cost, data privacy, and model reliability hinder their adoption. In this study, we propose a collaborative framework that enhances the performance of open-source LMMs for eGFR forecasting while generating clinically meaningful explanations. The framework incorporates visual knowledge transfer, abductive reasoning, and a short-term memory mechanism to enhance prediction accuracy and interpretability. Experimental results show that the proposed framework achieves predictive performance and interpretability comparable to proprietary models. It also provides plausible clinical reasoning processes behind each prediction. Our method sheds new light on building AI systems for healthcare that combine predictive accuracy with clinically grounded interpretability. 

---
# SAEL: Leveraging Large Language Models with Adaptive Mixture-of-Experts for Smart Contract Vulnerability Detection 

**Authors**: Lei Yu, Shiqi Cheng, Zhirong Huang, Jingyuan Zhang, Chenjie Shen, Junyi Lu, Li Yang, Fengjun Zhang, Jiajia Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.22371)  

**Abstract**: With the increasing security issues in blockchain, smart contract vulnerability detection has become a research focus. Existing vulnerability detection methods have their limitations: 1) Static analysis methods struggle with complex scenarios. 2) Methods based on specialized pre-trained models perform well on specific datasets but have limited generalization capabilities. In contrast, general-purpose Large Language Models (LLMs) demonstrate impressive ability in adapting to new vulnerability patterns. However, they often underperform on specific vulnerability types compared to methods based on specialized pre-trained models. We also observe that explanations generated by general-purpose LLMs can provide fine-grained code understanding information, contributing to improved detection performance.
Inspired by these observations, we propose SAEL, an LLM-based framework for smart contract vulnerability detection. We first design targeted prompts to guide LLMs in identifying vulnerabilities and generating explanations, which serve as prediction features. Next, we apply prompt-tuning on CodeT5 and T5 to process contract code and explanations, enhancing task-specific performance. To combine the strengths of each approach, we introduce an Adaptive Mixture-of-Experts architecture. This dynamically adjusts feature weights via a Gating Network, which selects relevant features using TopK filtering and Softmax normalization, and incorporates a Multi-Head Self-Attention mechanism to enhance cross-feature relationships. This design enables effective integration of LLM predictions, explanation features, and code features through gradient optimization. The loss function jointly considers both independent feature performance and overall weighted predictions. Experiments show that SAEL outperforms existing methods across various vulnerabilities. 

---
# Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems 

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany  

**Link**: [PDF](https://arxiv.org/pdf/2507.22239)  

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity. 

---
# From Articles to Code: On-Demand Generation of Core Algorithms from Scientific Publications 

**Authors**: Cameron S. Movassaghi, Amanda Momenzadeh, Jesse G. Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.22324)  

**Abstract**: Maintaining software packages imposes significant costs due to dependency management, bug fixes, and versioning. We show that rich method descriptions in scientific publications can serve as standalone specifications for modern large language models (LLMs), enabling on-demand code generation that could supplant human-maintained libraries. We benchmark state-of-the-art models (GPT-o4-mini-high, Gemini Pro 2.5, Claude Sonnet 4) by tasking them with implementing a diverse set of core algorithms drawn from original publications. Our results demonstrate that current LLMs can reliably reproduce package functionality with performance indistinguishable from conventional libraries. These findings foreshadow a paradigm shift toward flexible, on-demand code generation and away from static, human-maintained packages, which will result in reduced maintenance overhead by leveraging published articles as sufficient context for the automated implementation of analytical workflows. 

---
# IndoPref: A Multi-Domain Pairwise Preference Dataset for Indonesian 

**Authors**: Vanessa Rebecca Wiyono, David Anugraha, Ayu Purwarianti, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2507.22159)  

**Abstract**: Over 200 million people speak Indonesian, yet the language remains significantly underrepresented in preference-based research for large language models (LLMs). Most existing multilingual datasets are derived from English translations, often resulting in content that lacks cultural and linguistic authenticity. To address this gap, we introduce IndoPref, the first fully human-authored and multi-domain Indonesian preference dataset specifically designed to evaluate the naturalness and quality of LLM-generated text. All annotations are natively written in Indonesian and evaluated using Krippendorff's alpha, demonstrating strong inter-annotator agreement. Additionally, we benchmark the dataset across multiple LLMs and assess the output quality of each model. 

---
# From Cloud-Native to Trust-Native: A Protocol for Verifiable Multi-Agent Systems 

**Authors**: Muyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22077)  

**Abstract**: As autonomous agents powered by large language models (LLMs) proliferate in high-stakes domains -- from pharmaceuticals to legal workflows -- the challenge is no longer just intelligence, but verifiability. We introduce TrustTrack, a protocol that embeds structural guarantees -- verifiable identity, policy commitments, and tamper-resistant behavioral logs -- directly into agent infrastructure. This enables a new systems paradigm: trust-native autonomy. By treating compliance as a design constraint rather than post-hoc oversight, TrustTrack reframes how intelligent agents operate across organizations and jurisdictions. We present the protocol design, system requirements, and use cases in regulated domains such as pharmaceutical R&D, legal automation, and AI-native collaboration. We argue that the Cloud -> AI -> Agent -> Trust transition represents the next architectural layer for autonomous systems. 

---
# Promoting Online Safety by Simulating Unsafe Conversations with LLMs 

**Authors**: Owen Hoffman, Kangze Peng, Zehua You, Sajid Kamal, Sukrit Venkatagiri  

**Link**: [PDF](https://arxiv.org/pdf/2507.22267)  

**Abstract**: Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM. 

---
# Enhancing Jailbreak Attacks on LLMs via Persona Prompts 

**Authors**: Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22171)  

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at this https URL. 

---
# A Compute-Matched Re-Evaluation of TroVE on MATH 

**Authors**: Tobias Sesterhenn, Ian Berlot-Attwell, Janis Zenkner, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2507.22069)  

**Abstract**: Reusing established theorems and formulas is central to mathematical problem solving, serving as essential building blocks for tackling increasingly complex challenges. Recent work, TroVE, argues that code-generating Large Language Models (LLMs) can benefit similarly on the MATH benchmark by inducing and reusing higher-level toolboxes. By allocating computational budget across an ensemble of three modes -- directly generating code, creating tools, and reusing tools -- TroVE claims to outperform a PRIMITIVE baseline that only performs direct generation. However, recent analysis (Berlot-Attwell et al., 2024) casts doubt on these gains, noting that the tools created are often trivial or rarely reused, suggesting that improvements may stem from self-consistency or self-correction. In this work, we re-evaluate TroVE on MATH, analyze the impact of each of its modes, and show that its benefit does not come from these mechanisms, but simply from a higher computational budget spent for TroVE compared to PRIMITIVE. To this end, we also perform a small correction in the original implementation of TroVE's selection mechanism, boosting TroVE's performance on MATH by 3\% in accuracy. After matching for compute, the benefit of TroVE reduces to a marginal improvement of 1\%, suggesting that this toolbox approach does not provide a significant benefit on MATH. 

---
# RedCoder: Automated Multi-Turn Red Teaming for Code LLMs 

**Authors**: Wenjie Jacky Mo, Qin Liu, Xiaofei Wen, Dongwon Jung, Hadi Askari, Wenxuan Zhou, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22063)  

**Abstract**: Large Language Models (LLMs) for code generation (i.e., Code LLMs) have demonstrated impressive capabilities in AI-assisted software development and testing. However, recent studies have shown that these models are prone to generating vulnerable or even malicious code under adversarial settings. Existing red-teaming approaches rely on extensive human effort, limiting their scalability and practicality, and generally overlook the interactive nature of real-world AI-assisted programming, which often unfolds over multiple turns. To bridge these gaps, we present RedCoder, a red-teaming agent that engages victim models in multi-turn conversation to elicit vulnerable code. The pipeline to construct RedCoder begins with a multi-agent gaming process that simulates adversarial interactions, yielding a set of prototype conversations and an arsenal of reusable attack strategies. We then fine-tune an LLM on these prototype conversations to serve as the backbone of RedCoder. Once deployed, RedCoder autonomously engages Code LLMs in multi-turn conversations, dynamically retrieving relevant strategies from the arsenal to steer the dialogue toward vulnerability-inducing outputs. Experiments across multiple Code LLMs show that our approach outperforms prior single-turn and multi-turn red-team methods in inducing vulnerabilities in code generation, offering a scalable and effective tool for evaluating the security boundaries of modern code-generation systems. 

---
# Fuzzing: Randomness? Reasoning! Efficient Directed Fuzzing via Large Language Models 

**Authors**: Xiaotao Feng, Xiaogang Zhu, Kun Hu, Jincheng Wang, Yingjie Cao, Guang Gong, Jianfeng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22065)  

**Abstract**: Fuzzing is highly effective in detecting bugs due to the key contribution of randomness. However, randomness significantly reduces the efficiency of fuzzing, causing it to cost days or weeks to expose bugs. Even though directed fuzzing reduces randomness by guiding fuzzing towards target buggy locations, the dilemma of randomness still challenges directed fuzzers. Two critical components, which are seeds and mutators, contain randomness and are closely tied to the conditions required for triggering bugs. Therefore, to address the challenge of randomness, we propose to use large language models (LLMs) to remove the randomness in seeds and reduce the randomness in mutators. With their strong reasoning and code generation capabilities, LLMs can be used to generate reachable seeds that target pre-determined locations and to construct bug-specific mutators tailored for specific bugs. We propose RandLuzz, which integrates LLMs and directed fuzzing, to improve the quality of seeds and mutators, resulting in efficient bug exposure. RandLuzz analyzes function call chain or functionality to guide LLMs in generating reachable seeds. To construct bug-specific mutators, RandLuzz uses LLMs to perform bug analysis, obtaining information such as bug causes and mutation suggestions, which further help generate code that performs bug-specific mutations. We evaluate RandLuzz by comparing it with four state-of-the-art directed fuzzers, AFLGo, Beacon, WindRanger, and SelectFuzz. With RandLuzz-generated seeds, the fuzzers achieve an average speedup ranging from 2.1$\times$ to 4.8$\times$ compared to using widely-used initial seeds. Additionally, when evaluated on individual bugs, RandLuzz achieves up to a 2.7$\times$ speedup compared to the second-fastest exposure. On 8 bugs, RandLuzz can even expose them within 60 seconds. 

---
