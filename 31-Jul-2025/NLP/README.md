# Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning 

**Authors**: Kwesi Cobbina, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22887)  

**Abstract**: In-context learning (ICL) is a critical emerging capability of large language models (LLMs), enabling few-shot learning during inference by including a few demonstrations (demos) in the prompt. However, it has been found that ICL's performance can be sensitive to the choices of demos and their order. This paper investigates an unexplored new positional bias of ICL for the first time: we observe that the predictions and accuracy can drift drastically when the positions of demos, the system prompt, and the user message in LLM input are varied. We refer to this bias as DEMOS' POSITION IN PROMPT (DPP) bias. We design a systematic evaluation pipeline to study this type of positional bias across classification, question answering, summarization, and reasoning tasks. We introduce two metrics, ACCURACY-CHANGE and PREDICTION-CHANGE, to quantify net gains and output volatility induced by changes in the demos' position. Extensive experiments on ten LLMs from four open-source model families (QWEN, LLAMA3, MISTRAL, COHERE) verify that the bias significantly affects their accuracy and predictions: placing demos at the start of the prompt yields the most stable and accurate outputs with gains of up to +6 points. In contrast, placing demos at the end of the user message flips over 30\% of predictions without improving correctness on QA tasks. Smaller models are most affected by this sensitivity, though even large models remain marginally affected on complex tasks. 

---
# Beyond Natural Language Plans: Structure-Aware Planning for Query-Focused Table Summarization 

**Authors**: Weijia Zhang, Songgaojun Deng, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2507.22829)  

**Abstract**: Query-focused table summarization requires complex reasoning, often approached through step-by-step natural language (NL) plans. However, NL plans are inherently ambiguous and lack structure, limiting their conversion into executable programs like SQL and hindering scalability, especially for multi-table tasks. To address this, we propose a paradigm shift to structured representations. We introduce a new structured plan, TaSoF, inspired by formalism in traditional multi-agent systems, and a framework, SPaGe, that formalizes the reasoning process in three phases: 1) Structured Planning to generate TaSoF from a query, 2) Graph-based Execution to convert plan steps into SQL and model dependencies via a directed cyclic graph for parallel execution, and 3) Summary Generation to produce query-focused summaries. Our method explicitly captures complex dependencies and improves reliability. Experiments on three public benchmarks show that SPaGe consistently outperforms prior models in both single- and multi-table settings, demonstrating the advantages of structured representations for robust and scalable summarization. 

---
# DBLPLink 2.0 -- An Entity Linker for the DBLP Scholarly Knowledge Graph 

**Authors**: Debayan Banerjee, Tilahun Abedissa Taffa, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2507.22811)  

**Abstract**: In this work we present an entity linker for DBLP's 2025 version of RDF-based Knowledge Graph. Compared to the 2022 version, DBLP now considers publication venues as a new entity type called dblp:Stream. In the earlier version of DBLPLink, we trained KG-embeddings and re-rankers on a dataset to produce entity linkings. In contrast, in this work, we develop a zero-shot entity linker using LLMs using a novel method, where we re-rank candidate entities based on the log-probabilities of the "yes" token output at the penultimate layer of the LLM. 

---
# MASCA: LLM based-Multi Agents System for Credit Assessment 

**Authors**: Gautam Jajoo, Pranjal A Chitale, Saksham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.22758)  

**Abstract**: Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring. 

---
# Opportunities and Challenges of LLMs in Education: An NLP Perspective 

**Authors**: Sowmya Vajjala, Bashar Alhafni, Stefano Bannò, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.22753)  

**Abstract**: Interest in the role of large language models (LLMs) in education is increasing, considering the new opportunities they offer for teaching, learning, and assessment. In this paper, we examine the impact of LLMs on educational NLP in the context of two main application scenarios: {\em assistance} and {\em assessment}, grounding them along the four dimensions -- reading, writing, speaking, and tutoring. We then present the new directions enabled by LLMs, and the key challenges to address. We envision that this holistic overview would be useful for NLP researchers and practitioners interested in exploring the role of LLMs in developing language-focused and NLP-enabled educational applications of the future. 

---
# CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset 

**Authors**: Jindřich Libovický, Jindřich Helcl, Andrei Manea, Gianluca Vico  

**Link**: [PDF](https://arxiv.org/pdf/2507.22752)  

**Abstract**: We introduce a benchmark for open-ended regional question answering that encompasses both textual and visual modalities. We also provide strong baselines using state-of-the-art large language models (LLMs). Our dataset consists of manually curated questions and answers grounded in Wikipedia, created by native speakers from Czechia, Slovakia, and Ukraine, with accompanying English translations. It includes both purely textual questions and those requiring visual understanding. As a baseline, we evaluate state-of-the-art LLMs through prompting and complement this with human judgments of answer correctness. Using these human evaluations, we analyze the reliability of existing automatic evaluation metrics. Our baseline results highlight a significant gap in regional knowledge among current LLMs. Moreover, apart from LLM-based evaluation, there is minimal correlation between automated metrics and human judgment. We release this dataset as a resource to (1) assess regional knowledge in LLMs, (2) study cross-lingual generation consistency in a challenging setting, and (3) advance the development of evaluation metrics for open-ended question answering. 

---
# Reducing Hallucinations in Summarization via Reinforcement Learning with Entity Hallucination Index 

**Authors**: Praveenkumar Katwe, Rakesh Chandra, Balabantaray Kali, Prasad Vittala  

**Link**: [PDF](https://arxiv.org/pdf/2507.22744)  

**Abstract**: Reducing hallucinations in abstractive summarization remains a critical challenge for deploying language models (LMs) in real-world settings. In this work, we introduce a rewarddriven fine-tuning framework that explicitly optimizes for Entity Hallucination Index (EHI), a metric designed to quantify the presence, correctness, and grounding of named entities in generated summaries. Given a corpus of meeting transcripts, we first generate baseline summaries using a pre-trained LM and compute EHI scores via automatic entity extraction and matching. We then apply reinforcement learning to fine-tune the model parameters, using EHI as a reward signal to bias generation toward entity-faithful outputs. Our approach does not rely on human-written factuality annotations, enabling scalable fine-tuning. Experiments demonstrate consistent improvements in EHI across datasets, with qualitative analysis revealing a significant reduction in entity-level hallucinations without degradation in fluency or informativeness. We release a reproducible Colab pipeline, facilitating further research on hallucination-aware model fine-tuning using lightweight, hallucintion metrics like EHI. 

---
# Resource-Efficient Adaptation of Large Language Models for Text Embeddings via Prompt Engineering and Contrastive Fine-tuning 

**Authors**: Benedikt Roth, Stephan Rappensperger, Tianming Qiu, Hamza Imamović, Julian Wörmann, Hao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22729)  

**Abstract**: Large Language Models (LLMs) have become a cornerstone in Natural Language Processing (NLP), achieving impressive performance in text generation. Their token-level representations capture rich, human-aligned semantics. However, pooling these vectors into a text embedding discards crucial information. Nevertheless, many non-generative downstream tasks, such as clustering, classification, or retrieval, still depend on accurate and controllable sentence- or document-level embeddings. We explore several adaptation strategies for pre-trained, decoder-only LLMs: (i) various aggregation techniques for token embeddings, (ii) task-specific prompt engineering, and (iii) text-level augmentation via contrastive fine-tuning. Combining these components yields state-of-the-art performance on the English clustering track of the Massive Text Embedding Benchmark (MTEB). An analysis of the attention map further shows that fine-tuning shifts focus from prompt tokens to semantically relevant words, indicating more effective compression of meaning into the final hidden state. Our experiments demonstrate that LLMs can be effectively adapted as text embedding models through a combination of prompt engineering and resource-efficient contrastive fine-tuning on synthetically generated positive pairs. 

---
# Investigating Hallucination in Conversations for Low Resource Languages 

**Authors**: Amit Das, Md. Najib Hasan, Souvika Sarkar, Zheng Zhang, Fatemeh Jamshidi, Tathagata Bhattacharya, Nilanjana Raychawdhury, Dongji Feng, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2507.22720)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in generating text that closely resemble human writing. However, they often generate factually incorrect statements, a problem typically referred to as 'hallucination'. Addressing hallucination is crucial for enhancing the reliability and effectiveness of LLMs. While much research has focused on hallucinations in English, our study extends this investigation to conversational data in three languages: Hindi, Farsi, and Mandarin. We offer a comprehensive analysis of a dataset to examine both factual and linguistic errors in these languages for GPT-3.5, GPT-4o, Llama-3.1, Gemma-2.0, DeepSeek-R1 and Qwen-3. We found that LLMs produce very few hallucinated responses in Mandarin but generate a significantly higher number of hallucinations in Hindi and Farsi. 

---
# From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs 

**Authors**: Jie He, Victor Gutierrez Basulto, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22716)  

**Abstract**: Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: this https URL. 

---
# Listening to the Unspoken: Exploring 365 Aspects of Multimodal Interview Performance Assessment 

**Authors**: Jia Li, Yang Wang, Wenhao Qian, Zhenzhen Hu, Richang Hong, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22676)  

**Abstract**: Interview performance assessment is essential for determining candidates' suitability for professional positions. To ensure holistic and fair evaluations, we propose a novel and comprehensive framework that explores ``365'' aspects of interview performance by integrating \textit{three} modalities (video, audio, and text), \textit{six} responses per candidate, and \textit{five} key evaluation dimensions. The framework employs modality-specific feature extractors to encode heterogeneous data streams and subsequently fused via a Shared Compression Multilayer Perceptron. This module compresses multimodal embeddings into a unified latent space, facilitating efficient feature interaction. To enhance prediction robustness, we incorporate a two-level ensemble learning strategy: (1) independent regression heads predict scores for each response, and (2) predictions are aggregated across responses using a mean-pooling mechanism to produce final scores for the five target dimensions. By listening to the unspoken, our approach captures both explicit and implicit cues from multimodal data, enabling comprehensive and unbiased assessments. Achieving a multi-dimensional average MSE of 0.1824, our framework secured first place in the AVI Challenge 2025, demonstrating its effectiveness and robustness in advancing automated and multimodal interview performance assessment. The full implementation is available at this https URL. 

---
# Multilingual Political Views of Large Language Models: Identification and Steering 

**Authors**: Daniil Gurgurov, Katharina Trinley, Ivan Vykopal, Josef van Genabith, Simon Ostermann, Roberto Zamparelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.22623)  

**Abstract**: Large language models (LLMs) are increasingly used in everyday tools and applications, raising concerns about their potential influence on political views. While prior research has shown that LLMs often exhibit measurable political biases--frequently skewing toward liberal or progressive positions--key gaps remain. Most existing studies evaluate only a narrow set of models and languages, leaving open questions about the generalizability of political biases across architectures, scales, and multilingual settings. Moreover, few works examine whether these biases can be actively controlled.
In this work, we address these gaps through a large-scale study of political orientation in modern open-source instruction-tuned LLMs. We evaluate seven models, including LLaMA-3.1, Qwen-3, and Aya-Expanse, across 14 languages using the Political Compass Test with 11 semantically equivalent paraphrases per statement to ensure robust measurement. Our results reveal that larger models consistently shift toward libertarian-left positions, with significant variations across languages and model families. To test the manipulability of political stances, we utilize a simple center-of-mass activation intervention technique and show that it reliably steers model responses toward alternative ideological positions across multiple languages. Our code is publicly available at this https URL. 

---
# Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation 

**Authors**: Daniil Gurgurov, Katharina Trinley, Yusser Al Ghussin, Tanja Baeumel, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.22608)  

**Abstract**: Large language models (LLMs) exhibit strong multilingual abilities, yet the neural mechanisms behind language-specific processing remain unclear. We analyze language-specific neurons in Llama-3.1-8B, Mistral-Nemo-12B, and Aya-Expanse-8B & 32B across 21 typologically diverse languages, identifying neurons that control language behavior. Using the Language Activation Probability Entropy (LAPE) method, we show that these neurons cluster in deeper layers, with non-Latin scripts showing greater specialization. Related languages share overlapping neurons, reflecting internal representations of linguistic proximity.
Through language arithmetics, i.e. systematic activation addition and multiplication, we steer models to deactivate unwanted languages and activate desired ones, outperforming simpler replacement approaches. These interventions effectively guide behavior across five multilingual tasks: language forcing, translation, QA, comprehension, and NLI. Manipulation is more successful for high-resource languages, while typological similarity improves effectiveness. We also demonstrate that cross-lingual neuron steering enhances downstream performance and reveal internal "fallback" mechanisms for language selection when neurons are progressively deactivated. Our code is made publicly available at this https URL. 

---
# BALSAM: A Platform for Benchmarking Arabic Large Language Models 

**Authors**: Rawan Al-Matham, Kareem Darwish, Raghad Al-Rasheed, Waad Alshammari, Muneera Alhoshan, Amal Almazrua, Asma Al Wazrah, Mais Alheraki, Firoj Alam, Preslav Nakov, Norah Alzahrani, Eman alBilali, Nizar Habash, Abdelrahman El-Sheikh, Muhammad Elmallah, Haonan Li, Hamdy Mubarak, Mohamed Anwar, Zaid Alyafeai, Ahmed Abdelali, Nora Altwairesh, Maram Hasanain, Abdulmohsen Al Thubaity, Shady Shehata, Bashar Alhafni, Injy Hamed, Go Inoue, Khalid Elmadani, Ossama Obeid, Fatima Haouari, Tamer Elsayed, Emad Alghamdi, Khalid Almubarak, Saied Alshahrani, Ola Aljarrah, Safa Alajlan, Areej Alshaqarawi, Maryam Alshihri, Sultana Alghurabi, Atikah Alzeghayer, Afrah Altamimi, Abdullah Alfaifi, Abdulrahman AlOsaimy  

**Link**: [PDF](https://arxiv.org/pdf/2507.22603)  

**Abstract**: The impressive advancement of Large Language Models (LLMs) in English has not been matched across all languages. In particular, LLM performance in Arabic lags behind, due to data scarcity, linguistic diversity of Arabic and its dialects, morphological complexity, etc. Progress is further hindered by the quality of Arabic benchmarks, which typically rely on static, publicly available data, lack comprehensive task coverage, or do not provide dedicated platforms with blind test sets. This makes it challenging to measure actual progress and to mitigate data contamination. Here, we aim to bridge these gaps. In particular, we introduce BALSAM, a comprehensive, community-driven benchmark aimed at advancing Arabic LLM development and evaluation. It includes 78 NLP tasks from 14 broad categories, with 52K examples divided into 37K test and 15K development, and a centralized, transparent platform for blind evaluation. We envision BALSAM as a unifying platform that sets standards and promotes collaborative research to advance Arabic LLM capabilities. 

---
# Unveiling the Influence of Amplifying Language-Specific Neurons 

**Authors**: Inaya Rahmanisa, Lyzander Marciano Andrylie, Krisna Mahardika Ihsani, Alfan Farizki Wicaksono, Haryo Akbarianto Wibowo, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2507.22581)  

**Abstract**: Language-specific neurons in LLMs that strongly correlate with individual languages have been shown to influence model behavior by deactivating them. However, their role in amplification remains underexplored. This work investigates the effect of amplifying language-specific neurons through interventions across 18 languages, including low-resource ones, using three models primarily trained in different languages. We compare amplification factors by their effectiveness in steering to the target language using a proposed Language Steering Shift (LSS) evaluation score, then evaluate it on downstream tasks: commonsense reasoning (XCOPA, XWinograd), knowledge (Include), and translation (FLORES). The optimal amplification factors effectively steer output toward nearly all tested languages. Intervention using this factor on downstream tasks improves self-language performance in some cases but generally degrades cross-language results. These findings highlight the effect of language-specific neurons in multilingual behavior, where amplification can be beneficial especially for low-resource languages, but provides limited advantage for cross-lingual transfer. 

---
# Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs 

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22564)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems. 

---
# ControlMed: Adding Reasoning Control to Medical Language Model 

**Authors**: Sung-Min Lee, Siyoon Lee, Juyeon Kim, Kyungmin Roh  

**Link**: [PDF](https://arxiv.org/pdf/2507.22545)  

**Abstract**: Reasoning Large Language Models (LLMs) with enhanced accuracy and explainability are increasingly being adopted in the medical domain, as the life-critical nature of clinical decision-making demands reliable support. Despite these advancements, existing reasoning LLMs often generate unnecessarily lengthy reasoning processes, leading to significant computational overhead and response latency. These limitations hinder their practical deployment in real-world clinical environments. To address these challenges, we introduce \textbf{ControlMed}, a medical language model that enables users to actively control the length of the reasoning process at inference time through fine-grained control markers. ControlMed is trained through a three-stage pipeline: 1) pre-training on a large-scale synthetic medical instruction dataset covering both \textit{direct} and \textit{reasoning responses}; 2) supervised fine-tuning with multi-length reasoning data and explicit length-control markers; and 3) reinforcement learning with model-based reward signals to enhance factual accuracy and response quality. Experimental results on a variety of English and Korean medical benchmarks demonstrate that our model achieves similar or better performance compared to state-of-the-art models. Furthermore, users can flexibly balance reasoning accuracy and computational efficiency by controlling the reasoning length as needed. These findings demonstrate that ControlMed is a practical and adaptable solution for clinical question answering and medical information analysis. 

---
# A Benchmark Dataset and Evaluation Framework for Vietnamese Large Language Models in Customer Support 

**Authors**: Long S. T. Nguyen, Truong P. Hua, Thanh M. Nguyen, Toan Q. Pham, Nam K. Ngo, An X. Nguyen, Nghi D. M. Pham, Nghia H. Nguyen, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22542)  

**Abstract**: With the rapid growth of Artificial Intelligence, Large Language Models (LLMs) have become essential for Question Answering (QA) systems, improving efficiency and reducing human workload in customer service. The emergence of Vietnamese LLMs (ViLLMs) highlights lightweight open-source models as a practical choice for their accuracy, efficiency, and privacy benefits. However, domain-specific evaluations remain limited, and the absence of benchmark datasets reflecting real customer interactions makes it difficult for enterprises to select suitable models for support applications. To address this gap, we introduce the Customer Support Conversations Dataset (CSConDa), a curated benchmark of over 9,000 QA pairs drawn from real interactions with human advisors at a large Vietnamese software company. Covering diverse topics such as pricing, product availability, and technical troubleshooting, CSConDa provides a representative basis for evaluating ViLLMs in practical scenarios. We further present a comprehensive evaluation framework, benchmarking 11 lightweight open-source ViLLMs on CSConDa with both automatic metrics and syntactic analysis to reveal model strengths, weaknesses, and linguistic patterns. This study offers insights into model behavior, explains performance differences, and identifies key areas for improvement, supporting the development of next-generation ViLLMs. By establishing a robust benchmark and systematic evaluation, our work enables informed model selection for customer service QA and advances research on Vietnamese LLMs. The dataset is publicly available at this https URL. 

---
# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records 

**Authors**: Dongchen Li, Jitao Liang, Wei Li, Xiaoyu Wang, Longbing Cao, Kun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22533)  

**Abstract**: Large Language Models (LLMs) hold significant promise for improving clinical decision support and reducing physician burnout by synthesizing complex, longitudinal cancer Electronic Health Records (EHRs). However, their implementation in this critical field faces three primary challenges: the inability to effectively process the extensive length and multilingual nature of patient records for accurate temporal analysis; a heightened risk of clinical hallucination, as conventional grounding techniques such as Retrieval-Augmented Generation (RAG) do not adequately incorporate process-oriented clinical guidelines; and unreliable evaluation metrics that hinder the validation of AI systems in oncology. To address these issues, we propose CliCARE, a framework for Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. The framework operates by transforming unstructured, longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range dependencies, and then grounding the decision support process by aligning these real-world patient trajectories with a normative guideline knowledge graph. This approach provides oncologists with evidence-grounded decision support by generating a high-fidelity clinical summary and an actionable recommendation. We validated our framework using large-scale, longitudinal data from a private Chinese cancer dataset and the public English MIMIC-IV dataset. In these diverse settings, CliCARE significantly outperforms strong baselines, including leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of our results is supported by a robust evaluation protocol, which demonstrates a high correlation with assessments made by expert oncologists. 

---
# SLM-SQL: An Exploration of Small Language Models for Text-to-SQL 

**Authors**: Lei Sheng, Shuai-Shuai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22478)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in translating natural language questions into SQL queries (Text-to-SQL). In contrast, small language models (SLMs) ranging from 0.5B to 1.5B parameters currently underperform on Text-to-SQL tasks due to their limited logical reasoning capabilities. However, SLMs offer inherent advantages in inference speed and suitability for edge deployment. To explore their potential in Text-to-SQL applications, we leverage recent advancements in post-training techniques. Specifically, we used the open-source SynSQL-2.5M dataset to construct two derived datasets: SynSQL-Think-916K for SQL generation and SynSQL-Merge-Think-310K for SQL merge revision. We then applied supervised fine-tuning and reinforcement learning-based post-training to the SLM, followed by inference using a corrective self-consistency approach. Experimental results validate the effectiveness and generalizability of our method, SLM-SQL. On the BIRD development set, the five evaluated models achieved an average improvement of 31.4 points. Notably, the 0.5B model reached 56.87\% execution accuracy (EX), while the 1.5B model achieved 67.08\% EX. We will release our dataset, model, and code to github: this https URL. 

---
# IFEvalCode: Controlled Code Generation 

**Authors**: Jian Yang, Wei Zhang, Shukai Liu, Linzheng Chai, Yingshui Tan, Jiaheng Liu, Ge Zhang, Wangchunshu Zhou, Guanglin Niu, Zhoujun Li, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22462)  

**Abstract**: Code large language models (Code LLMs) have made significant progress in code generation by translating natural language descriptions into functional code; however, real-world applications often demand stricter adherence to detailed requirements such as coding style, line count, and structural constraints, beyond mere correctness. To address this, the paper introduces forward and backward constraints generation to improve the instruction-following capabilities of Code LLMs in controlled code generation, ensuring outputs align more closely with human-defined guidelines. The authors further present IFEvalCode, a multilingual benchmark comprising 1.6K test samples across seven programming languages (Python, Java, JavaScript, TypeScript, Shell, C++, and C#), with each sample featuring both Chinese and English queries. Unlike existing benchmarks, IFEvalCode decouples evaluation into two metrics: correctness (Corr.) and instruction-following (Instr.), enabling a more nuanced assessment. Experiments on over 40 LLMs reveal that closed-source models outperform open-source ones in controllable code generation and highlight a significant gap between the models' ability to generate correct code versus code that precisely follows instructions. 

---
# What is an "Abstract Reasoner"? Revisiting Experiments and Arguments about Large Language Models 

**Authors**: Tian Yun, Chen Sun, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.22457)  

**Abstract**: Recent work has argued that large language models (LLMs) are not "abstract reasoners", citing their poor zero-shot performance on a variety of challenging tasks as evidence. We revisit these experiments in order to add nuance to the claim. First, we show that while LLMs indeed perform poorly in a zero-shot setting, even tuning a small subset of parameters for input encoding can enable near-perfect performance. However, we also show that this finetuning does not necessarily transfer across datasets. We take this collection of empirical results as an invitation to (re-)open the discussion of what it means to be an "abstract reasoner", and why it matters whether LLMs fit the bill. 

---
# Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance 

**Authors**: Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, Slim Frikha  

**Link**: [PDF](https://arxiv.org/pdf/2507.22448)  

**Abstract**: In this report, we introduce Falcon-H1, a new series of large language models (LLMs) featuring hybrid architecture designs optimized for both high performance and efficiency across diverse use cases. Unlike earlier Falcon models built solely on Transformer or Mamba architectures, Falcon-H1 adopts a parallel hybrid approach that combines Transformer-based attention with State Space Models (SSMs), known for superior long-context memory and computational efficiency. We systematically revisited model design, data strategy, and training dynamics, challenging conventional practices in the field. Falcon-H1 is released in multiple configurations, including base and instruction-tuned variants at 0.5B, 1.5B, 1.5B-deep, 3B, 7B, and 34B parameters. Quantized instruction-tuned models are also available, totaling over 30 checkpoints on Hugging Face Hub. Falcon-H1 models demonstrate state-of-the-art performance and exceptional parameter and training efficiency. The flagship Falcon-H1-34B matches or outperforms models up to 70B scale, such as Qwen3-32B, Qwen2.5-72B, and Llama3.3-70B, while using fewer parameters and less data. Smaller models show similar trends: the Falcon-H1-1.5B-Deep rivals current leading 7B-10B models, and Falcon-H1-0.5B performs comparably to typical 7B models from 2024. These models excel across reasoning, mathematics, multilingual tasks, instruction following, and scientific knowledge. With support for up to 256K context tokens and 18 languages, Falcon-H1 is suitable for a wide range of applications. All models are released under a permissive open-source license, underscoring our commitment to accessible and impactful AI research. 

---
# AI-generated stories favour stability over change: homogeneity and cultural stereotyping in narratives generated by gpt-4o-mini 

**Authors**: Jill Walker Rettberg, Hermann Wigers  

**Link**: [PDF](https://arxiv.org/pdf/2507.22445)  

**Abstract**: Can a language model trained largely on Anglo-American texts generate stories that are culturally relevant to other nationalities? To find out, we generated 11,800 stories - 50 for each of 236 countries - by sending the prompt "Write a 1500 word potential {demonym} story" to OpenAI's model gpt-4o-mini. Although the stories do include surface-level national symbols and themes, they overwhelmingly conform to a single narrative plot structure across countries: a protagonist lives in or returns home to a small town and resolves a minor conflict by reconnecting with tradition and organising community events. Real-world conflicts are sanitised, romance is almost absent, and narrative tension is downplayed in favour of nostalgia and reconciliation. The result is a narrative homogenisation: an AI-generated synthetic imaginary that prioritises stability above change and tradition above growth. We argue that the structural homogeneity of AI-generated narratives constitutes a distinct form of AI bias, a narrative standardisation that should be acknowledged alongside the more familiar representational bias. These findings are relevant to literary studies, narratology, critical AI studies, NLP research, and efforts to improve the cultural alignment of generative AI. 

---
# NeedleChain: Measuring Intact Long-Context Reasoning Capability of Large Language Models 

**Authors**: Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22411)  

**Abstract**: The Needle-in-a-Haystack (NIAH) benchmark is widely used to evaluate Large Language Models' (LLMs) ability to understand long contexts (LC). It evaluates the capability to identify query-relevant context within extensive query-irrelevant passages. Although this method serves as a widely accepted standard for evaluating long-context understanding, our findings suggest it may overestimate the true LC capability of LLMs. We demonstrate that even state-of-the-art models such as GPT-4o struggle to intactly incorporate given contexts made up of solely query-relevant ten sentences. In response, we introduce a novel benchmark, \textbf{NeedleChain}, where the context consists entirely of query-relevant information, requiring the LLM to fully grasp the input to answer correctly. Our benchmark allows for flexible context length and reasoning order, offering a more comprehensive analysis of LLM performance. Additionally, we propose an extremely simple yet compelling strategy to improve LC understanding capability of LLM: ROPE Contraction. Our experiments with various advanced LLMs reveal a notable disparity between their ability to process large contexts and their capacity to fully understand them. Source code and datasets are available at this https URL 

---
# Question Generation for Assessing Early Literacy Reading Comprehension 

**Authors**: Xiaocheng Yang, Sumuk Shashidhar, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2507.22410)  

**Abstract**: Assessment of reading comprehension through content-based interactions plays an important role in the reading acquisition process. In this paper, we propose a novel approach for generating comprehension questions geared to K-2 English learners. Our method ensures complete coverage of the underlying material and adaptation to the learner's specific proficiencies, and can generate a large diversity of question types at various difficulty levels to ensure a thorough evaluation. We evaluate the performance of various language models in this framework using the FairytaleQA dataset as the source material. Eventually, the proposed approach has the potential to become an important part of autonomous AI-driven English instructors. 

---
# PATENTWRITER: A Benchmarking Study for Patent Drafting with LLMs 

**Authors**: Homaira Huda Shomee, Suman Kalyan Maity, Sourav Medya  

**Link**: [PDF](https://arxiv.org/pdf/2507.22387)  

**Abstract**: Large language models (LLMs) have emerged as transformative approaches in several important fields. This paper aims for a paradigm shift for patent writing by leveraging LLMs to overcome the tedious patent-filing process. In this work, we present PATENTWRITER, the first unified benchmarking framework for evaluating LLMs in patent abstract generation. Given the first claim of a patent, we evaluate six leading LLMs -- including GPT-4 and LLaMA-3 -- under a consistent setup spanning zero-shot, few-shot, and chain-of-thought prompting strategies to generate the abstract of the patent. Our benchmark PATENTWRITER goes beyond surface-level evaluation: we systematically assess the output quality using a comprehensive suite of metrics -- standard NLP measures (e.g., BLEU, ROUGE, BERTScore), robustness under three types of input perturbations, and applicability in two downstream patent classification and retrieval tasks. We also conduct stylistic analysis to assess length, readability, and tone. Experimental results show that modern LLMs can generate high-fidelity and stylistically appropriate patent abstracts, often surpassing domain-specific baselines. Our code and dataset are open-sourced to support reproducibility and future research. 

---
# Traits Run Deep: Enhancing Personality Assessment via Psychology-Guided LLM Representations and Multimodal Apparent Behaviors 

**Authors**: Jia Li, Yichao He, Jiacheng Xu, Tianhao Luo, Zhenzhen Hu, Richang Hong, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22367)  

**Abstract**: Accurate and reliable personality assessment plays a vital role in many fields, such as emotional intelligence, mental health diagnostics, and personalized education. Unlike fleeting emotions, personality traits are stable, often subconsciously leaked through language, facial expressions, and body behaviors, with asynchronous patterns across modalities. It was hard to model personality semantics with traditional superficial features and seemed impossible to achieve effective cross-modal understanding. To address these challenges, we propose a novel personality assessment framework called \textit{\textbf{Traits Run Deep}}. It employs \textit{\textbf{psychology-informed prompts}} to elicit high-level personality-relevant semantic representations. Besides, it devises a \textit{\textbf{Text-Centric Trait Fusion Network}} that anchors rich text semantics to align and integrate asynchronous signals from other modalities. To be specific, such fusion module includes a Chunk-Wise Projector to decrease dimensionality, a Cross-Modal Connector and a Text Feature Enhancer for effective modality fusion and an ensemble regression head to improve generalization in data-scarce situations. To our knowledge, we are the first to apply personality-specific prompts to guide large language models (LLMs) in extracting personality-aware semantics for improved representation quality. Furthermore, extracting and fusing audio-visual apparent behavior features further improves the accuracy. Experimental results on the AVI validation set have demonstrated the effectiveness of the proposed components, i.e., approximately a 45\% reduction in mean squared error (MSE). Final evaluations on the test set of the AVI Challenge 2025 confirm our method's superiority, ranking first in the Personality Assessment track. The source code will be made available at this https URL. 

---
# A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers 

**Authors**: Roxana Petcu, Samarth Bhargav, Maarten de Rijke, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2507.22337)  

**Abstract**: Understanding and solving complex reasoning tasks is vital for addressing the information needs of a user. Although dense neural models learn contextualised embeddings, they still underperform on queries containing negation. To understand this phenomenon, we study negation in both traditional neural information retrieval and LLM-based models. We (1) introduce a taxonomy of negation that derives from philosophical, linguistic, and logical definitions; (2) generate two benchmark datasets that can be used to evaluate the performance of neural information retrieval models and to fine-tune models for a more robust performance on negation; and (3) propose a logic-based classification mechanism that can be used to analyze the performance of retrieval models on existing datasets. Our taxonomy produces a balanced data distribution over negation types, providing a better training setup that leads to faster convergence on the NevIR dataset. Moreover, we propose a classification schema that reveals the coverage of negation types in existing datasets, offering insights into the factors that might affect the generalization of fine-tuned models on negation. 

---
# Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations 

**Authors**: Galo Castillo-López, Gaël de Chalendar, Nasredine Semmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.22289)  

**Abstract**: Intent recognition is a fundamental component in task-oriented dialogue systems (TODS). Determining user intents and detecting whether an intent is Out-of-Scope (OOS) is crucial for TODS to provide reliable responses. However, traditional TODS require large amount of annotated data. In this work we propose a hybrid approach to combine BERT and LLMs in zero and few-shot settings to recognize intents and detect OOS utterances. Our approach leverages LLMs generalization power and BERT's computational efficiency in such scenarios. We evaluate our method on multi-party conversation corpora and observe that sharing information from BERT outputs to LLMs leads to system performance improvement. 

---
# Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs 

**Authors**: Supantho Rakshit, Adele Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.22286)  

**Abstract**: The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs. 

---
# RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation 

**Authors**: Dongyub Jude Lee, Zhenyi Ye, Pengcheng He  

**Link**: [PDF](https://arxiv.org/pdf/2507.22219)  

**Abstract**: Preference-learning methods for machine translation (MT)--such as Direct Preference Optimization (DPO)--have achieved impressive gains but depend heavily on large, carefully curated triplet datasets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), a novel framework that removes reliance on static triplets by leveraging continuous, high-quality feedback from an external teacher model (GPT-4o). RLfR frames each translation step as a micro-tutorial: the actor generates a hypothesis, the teacher refines it, and the actor is rewarded based on how closely it aligns with the teacher's refinement. Guided by two complementary signals--(i) negative edit distance, promoting lexical and structural fidelity, and (ii) COMET score, ensuring semantic adequacy--the actor progressively learns to emulate the teacher, mirroring a human learning process through incremental, iterative improvement. On the FLORES-200 benchmark (English to and from German, Spanish, Chinese, Korean, and Japanese), RLfR consistently outperforms both MT-SFT and preference-based baselines, significantly improving COMET (semantic adequacy) and M-ETA (entity preservation) scores. 

---
# How Well Does First-Token Entropy Approximate Word Entropy as a Psycholinguistic Predictor? 

**Authors**: Christian Clark, Byung-Doh Oh, William Schuler  

**Link**: [PDF](https://arxiv.org/pdf/2507.22209)  

**Abstract**: Contextual entropy is a psycholinguistic measure capturing the anticipated difficulty of processing a word just before it is encountered. Recent studies have tested for entropy-related effects as a potential complement to well-known effects from surprisal. For convenience, entropy is typically estimated based on a language model's probability distribution over a word's first subword token. However, this approximation results in underestimation and potential distortion of true word entropy. To address this, we generate Monte Carlo (MC) estimates of word entropy that allow words to span a variable number of tokens. Regression experiments on reading times show divergent results between first-token and MC word entropy, suggesting a need for caution in using first-token approximations of contextual entropy. 

---
# The role of media memorability in facilitating startups' access to venture capital funding 

**Authors**: L. Toschi, S. Torrisi, A. Fronzetti Colladon  

**Link**: [PDF](https://arxiv.org/pdf/2507.22201)  

**Abstract**: Media reputation plays an important role in attracting venture capital investment. However, prior research has focused too narrowly on general media exposure, limiting our understanding of how media truly influences funding decisions. As informed decision-makers, venture capitalists respond to more nuanced aspects of media content. We introduce the concept of media memorability - the media's ability to imprint a startup's name in the memory of relevant investors. Using data from 197 UK startups in the micro and nanotechnology sector (funded between 1995 and 2004), we show that media memorability significantly influences investment outcomes. Our findings suggest that venture capitalists rely on detailed cues such as a startup's distinctiveness and connectivity within news semantic networks. This contributes to research on entrepreneurial finance and media legitimation. In practice, startups should go beyond frequent media mentions to strengthen brand memorability through more targeted, meaningful coverage highlighting their uniqueness and relevance within the broader industry conversation. 

---
# A Scalable Pipeline for Estimating Verb Frame Frequencies Using Large Language Models 

**Authors**: Adam M. Morgan, Adeen Flinker  

**Link**: [PDF](https://arxiv.org/pdf/2507.22187)  

**Abstract**: We present an automated pipeline for estimating Verb Frame Frequencies (VFFs), the frequency with which a verb appears in particular syntactic frames. VFFs provide a powerful window into syntax in both human and machine language systems, but existing tools for calculating them are limited in scale, accuracy, or accessibility. We use large language models (LLMs) to generate a corpus of sentences containing 476 English verbs. Next, by instructing an LLM to behave like an expert linguist, we had it analyze the syntactic structure of the sentences in this corpus. This pipeline outperforms two widely used syntactic parsers across multiple evaluation datasets. Furthermore, it requires far fewer resources than manual parsing (the gold-standard), thereby enabling rapid, scalable VFF estimation. Using the LLM parser, we produce a new VFF database with broader verb coverage, finer-grained syntactic distinctions, and explicit estimates of the relative frequencies of structural alternates commonly studied in psycholinguistics. The pipeline is easily customizable and extensible to new verbs, syntactic frames, and even other languages. We present this work as a proof of concept for automated frame frequency estimation, and release all code and data to support future research. 

---
# Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles 

**Authors**: Kimberly Le Truong, Riccardo Fogliato, Hoda Heidari, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22168)  

**Abstract**: Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations. 

---
# IndoPref: A Multi-Domain Pairwise Preference Dataset for Indonesian 

**Authors**: Vanessa Rebecca Wiyono, David Anugraha, Ayu Purwarianti, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2507.22159)  

**Abstract**: Over 200 million people speak Indonesian, yet the language remains significantly underrepresented in preference-based research for large language models (LLMs). Most existing multilingual datasets are derived from English translations, often resulting in content that lacks cultural and linguistic authenticity. To address this gap, we introduce IndoPref, the first fully human-authored and multi-domain Indonesian preference dataset specifically designed to evaluate the naturalness and quality of LLM-generated text. All annotations are natively written in Indonesian and evaluated using Krippendorff's alpha, demonstrating strong inter-annotator agreement. Additionally, we benchmark the dataset across multiple LLMs and assess the output quality of each model. 

---
# RecGPT Technical Report 

**Authors**: Chao Yi, Dian Chen, Gaoyang Guo, Jiakai Tang, Jian Wu, Jing Yu, Sunhao Dai, Wen Chen, Wenjun Yang, Yuning Jiang, Zhujin Gao, Bo Zheng, Chi Li, Dimin Wang, Dixuan Wang, Fan Li, Fan Zhang, Haibin Chen, Haozhuang Liu, Jialin Zhu, Jiamang Wang, Jiawei Wu, Jin Cui, Ju Huang, Kai Zhang, Kan Liu, Lang Tian, Liang Rao, Longbin Li, Lulu Zhao, Mao Zhang, Na He, Peiyang Wang, Qiqi Huang, Tao Luo, Wenbo Su, Xiaoxiao He, Xin Tong, Xu Chen, Xunke Xi, Yang Li, Yaxuan Wu, Yeqiu Yang, Yi Hu, Yinnan Song, Yuchen Li, Yujie Luo, Yujin Yuan, Yuliang Yan, Zhengyang Wang, Zhibo Xiao, Zhixin Ma, Zile Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22879)  

**Abstract**: Recommender systems are among the most impactful applications of artificial intelligence, serving as critical infrastructure connecting users, merchants, and platforms. However, most current industrial systems remain heavily reliant on historical co-occurrence patterns and log-fitting objectives, i.e., optimizing for past user interactions without explicitly modeling user intent. This log-fitting approach often leads to overfitting to narrow historical preferences, failing to capture users' evolving and latent interests. As a result, it reinforces filter bubbles and long-tail phenomena, ultimately harming user experience and threatening the sustainability of the whole recommendation ecosystem.
To address these challenges, we rethink the overall design paradigm of recommender systems and propose RecGPT, a next-generation framework that places user intent at the center of the recommendation pipeline. By integrating large language models (LLMs) into key stages of user interest mining, item retrieval, and explanation generation, RecGPT transforms log-fitting recommendation into an intent-centric process. To effectively align general-purpose LLMs to the above domain-specific recommendation tasks at scale, RecGPT incorporates a multi-stage training paradigm, which integrates reasoning-enhanced pre-alignment and self-training evolution, guided by a Human-LLM cooperative judge system. Currently, RecGPT has been fully deployed on the Taobao App. Online experiments demonstrate that RecGPT achieves consistent performance gains across stakeholders: users benefit from increased content diversity and satisfaction, merchants and the platform gain greater exposure and conversions. These comprehensive improvement results across all stakeholders validates that LLM-driven, intent-centric design can foster a more sustainable and mutually beneficial recommendation ecosystem. 

---
# GeoOutageKG: A Multimodal Geospatiotemporal Knowledge Graph for Multiresolution Power Outage Analysis 

**Authors**: Ethan Frakes, Yinghui Wu, Roger H. French, Mengjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22878)  

**Abstract**: Detecting, analyzing, and predicting power outages is crucial for grid risk assessment and disaster mitigation. Numerous outages occur each year, exacerbated by extreme weather events such as hurricanes. Existing outage data are typically reported at the county level, limiting their spatial resolution and making it difficult to capture localized patterns. However, it offers excellent temporal granularity. In contrast, nighttime light satellite image data provides significantly higher spatial resolution and enables a more comprehensive spatial depiction of outages, enhancing the accuracy of assessing the geographic extent and severity of power loss after disaster events. However, these satellite data are only available on a daily basis. Integrating spatiotemporal visual and time-series data sources into a unified knowledge representation can substantially improve power outage detection, analysis, and predictive reasoning. In this paper, we propose GeoOutageKG, a multimodal knowledge graph that integrates diverse data sources, including nighttime light satellite image data, high-resolution spatiotemporal power outage maps, and county-level timeseries outage reports in the U.S. We describe our method for constructing GeoOutageKG by aligning source data with a developed ontology, GeoOutageOnto. Currently, GeoOutageKG includes over 10.6 million individual outage records spanning from 2014 to 2024, 300,000 NTL images spanning from 2012 to 2024, and 15,000 outage maps. GeoOutageKG is a novel, modular and reusable semantic resource that enables robust multimodal data integration. We demonstrate its use through multiresolution analysis of geospatiotemporal power outages. 

---
# The Incomplete Bridge: How AI Research (Mis)Engages with Psychology 

**Authors**: Han Jiang, Pengda Wang, Xiaoyuan Yi, Xing Xie, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22847)  

**Abstract**: Social sciences have accumulated a rich body of theories and methodologies for investigating the human mind and behaviors, while offering valuable insights into the design and understanding of Artificial Intelligence (AI) systems. Focusing on psychology as a prominent case, this study explores the interdisciplinary synergy between AI and the field by analyzing 1,006 LLM-related papers published in premier AI venues between 2023 and 2025, along with the 2,544 psychology publications they cite. Through our analysis, we identify key patterns of interdisciplinary integration, locate the psychology domains most frequently referenced, and highlight areas that remain underexplored. We further examine how psychology theories/frameworks are operationalized and interpreted, identify common types of misapplication, and offer guidance for more effective incorporation. Our work provides a comprehensive map of interdisciplinary engagement between AI and psychology, thereby facilitating deeper collaboration and advancing AI systems. 

---
# Next Tokens Denoising for Speech Synthesis 

**Authors**: Yanqing Liu, Ruiqing Xue, Chong Zhang, Yufei Liu, Gang Wang, Bohan Li, Yao Qian, Lei He, Shujie Liu, Sheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22746)  

**Abstract**: While diffusion and autoregressive (AR) models have significantly advanced generative modeling, they each present distinct limitations. AR models, which rely on causal attention, cannot exploit future context and suffer from slow generation speeds. Conversely, diffusion models struggle with key-value (KV) caching. To overcome these challenges, we introduce Dragon-FM, a novel text-to-speech (TTS) design that unifies AR and flow-matching. This model processes 48 kHz audio codec tokens in chunks at a compact 12.5 tokens per second rate. This design enables AR modeling across chunks, ensuring global coherence, while parallel flow-matching within chunks facilitates fast iterative denoising. Consequently, the proposed model can utilize KV-cache across chunks and incorporate future context within each chunk. Furthermore, it bridges continuous and discrete feature modeling, demonstrating that continuous AR flow-matching can predict discrete tokens with finite scalar quantizers. This efficient codec and fast chunk-autoregressive architecture also makes the proposed model particularly effective for generating extended content. Experiment for demos of our work} on podcast datasets demonstrate its capability to efficiently generate high-quality zero-shot podcasts. 

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
# Pre-trained Models Perform the Best When Token Distributions Follow Zipf's Law 

**Authors**: Yanjin He, Qingkai Zeng, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22543)  

**Abstract**: Tokenization is a fundamental step in natural language processing (NLP) and other sequence modeling domains, where the choice of vocabulary size significantly impacts model performance. Despite its importance, selecting an optimal vocabulary size remains underexplored, typically relying on heuristics or dataset-specific choices. In this work, we propose a principled method for determining the vocabulary size by analyzing token frequency distributions through Zipf's law. We show that downstream task performance correlates with how closely token distributions follow power-law behavior, and that aligning with Zipfian scaling improves both model efficiency and effectiveness. Extensive experiments across NLP, genomics, and chemistry demonstrate that models consistently achieve peak performance when the token distribution closely adheres to Zipf's law, establishing Zipfian alignment as a robust and generalizable criterion for vocabulary size selection. 

---
# LLM-Crowdsourced: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models 

**Authors**: Qianhong Guo, Wei Xie, Xiaofang Cai, Enze Wang, Shuoyoucheng Ma, Kai Chen, Xiaofeng Wang, Baosheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22359)  

**Abstract**: Although large language models (LLMs) demonstrate remarkable capabilities across various tasks, evaluating their capabilities remains a challenging task. Existing evaluation methods suffer from issues such as data contamination, black-box operation, and subjective preference. These issues make it difficult to evaluate the LLMs' true capabilities comprehensively. To tackle these challenges, we propose a novel benchmark-free evaluation paradigm, LLM-Crowdsourced. It utilizes LLMs to generate questions, answer independently, and evaluate mutually. This method integrates four key evaluation criteria: dynamic, transparent, objective, and professional, which existing evaluation methods cannot satisfy simultaneously. Experiments on eight mainstream LLMs across mathematics and programming verify the advantages of our method in distinguishing LLM performance. Furthermore, our study reveals several novel findings that are difficult for traditional methods to detect, including but not limited to: (1) Gemini demonstrates the highest original and professional question-design capabilities among others; (2) Some LLMs exhibit ''memorization-based answering'' by misrecognizing questions as familiar ones with a similar structure; (3) LLM evaluation results demonstrate high consistency (robustness). 

---
# CoEx -- Co-evolving World-model and Exploration 

**Authors**: Minsoo Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22281)  

**Abstract**: Planning in modern LLM agents relies on the utilization of LLM as an internal world model, acquired during pretraining. However, existing agent designs fail to effectively assimilate new observations into dynamic updates of the world model. This reliance on the LLM's static internal world model is progressively prone to misalignment with the underlying true state of the world, leading to the generation of divergent and erroneous plans. We introduce a hierarchical agent architecture, CoEx, in which hierarchical state abstraction allows LLM planning to co-evolve with a dynamically updated model of the world. CoEx plans and interacts with the world by using LLM reasoning to orchestrate dynamic plans consisting of subgoals, and its learning mechanism continuously incorporates these subgoal experiences into a persistent world model in the form of a neurosymbolic belief state, comprising textual inferences and code-based symbolic memory. We evaluate our agent across a diverse set of agent scenarios involving rich environments and complex tasks including ALFWorld, PDDL, and Jericho. Our experiments show that CoEx outperforms existing agent paradigms in planning and exploration. 

---
# Explainability Through Systematicity: The Hard Systematicity Challenge for Artificial Intelligence 

**Authors**: Matthieu Queloz  

**Link**: [PDF](https://arxiv.org/pdf/2507.22197)  

**Abstract**: This paper argues that explainability is only one facet of a broader ideal that shapes our expectations towards artificial intelligence (AI). Fundamentally, the issue is to what extent AI exhibits systematicity--not merely in being sensitive to how thoughts are composed of recombinable constituents, but in striving towards an integrated body of thought that is consistent, coherent, comprehensive, and parsimoniously principled. This richer conception of systematicity has been obscured by the long shadow of the "systematicity challenge" to connectionism, according to which network architectures are fundamentally at odds with what Fodor and colleagues termed "the systematicity of thought." I offer a conceptual framework for thinking about "the systematicity of thought" that distinguishes four senses of the phrase. I use these distinctions to defuse the perceived tension between systematicity and connectionism and show that the conception of systematicity that historically shaped our sense of what makes thought rational, authoritative, and scientific is more demanding than the Fodorian notion. To determine whether we have reason to hold AI models to this ideal of systematicity, I then argue, we must look to the rationales for systematization and explore to what extent they transfer to AI models. I identify five such rationales and apply them to AI. This brings into view the "hard systematicity challenge." However, the demand for systematization itself needs to be regulated by the rationales for systematization. This yields a dynamic understanding of the need to systematize thought, which tells us how systematic we need AI models to be and when. 

---
# Strategic Deflection: Defending LLMs from Logit Manipulation 

**Authors**: Yassine Rachidy, Jihad Rbaiti, Youssef Hmamouche, Faissal Sehbaoui, Amal El Fallah Seghrouchni  

**Link**: [PDF](https://arxiv.org/pdf/2507.22160)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats. 

---
# Prompt Optimization and Evaluation for LLM Automated Red Teaming 

**Authors**: Michael Freenor, Lauren Alvarez, Milton Leal, Lily Smith, Joel Garrett, Yelyzaveta Husieva, Madeline Woodruff, Ryan Miller, Erich Kummerfeld, Rafael Medeiros, Sander Schulhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.22133)  

**Abstract**: Applications that use Large Language Models (LLMs) are becoming widespread, making the identification of system vulnerabilities increasingly important. Automated Red Teaming accelerates this effort by using an LLM to generate and execute attacks against target systems. Attack generators are evaluated using the Attack Success Rate (ASR) the sample mean calculated over the judgment of success for each attack. In this paper, we introduce a method for optimizing attack generator prompts that applies ASR to individual attacks. By repeating each attack multiple times against a randomly seeded target, we measure an attack's discoverability the expectation of the individual attack success. This approach reveals exploitable patterns that inform prompt optimization, ultimately enabling more robust evaluation and refinement of generators. 

---
# CodeEvo: Interaction-Driven Synthesis of Code-centric Data through Hybrid and Iterative Feedback 

**Authors**: Qiushi Sun, Jinyang Gong, Lei Li, Qipeng Guo, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22080)  

**Abstract**: Acquiring high-quality instruction-code pairs is essential for training Large Language Models (LLMs) for code generation. Manually curated data is expensive and inherently limited in scale, motivating the development of code-centric synthesis methods. Yet, current approaches either focus on augmenting existing code or rely on predefined heuristics, both lacking rigorous data validation, which results in synthetic data that is ungrounded, repetitive, or overly simplistic. Inspired by collaborative programming practices, we propose CodeEvo, a framework that synthesizes code data through iterative interactions between two LLM agents: a Coder, which generates candidate code and test cases based on given instructions, and a Reviewer, which guides the synthesis process by producing new instructions and feedback. We further introduce a hybrid feedback mechanism that combines compiler determinism with the generative flexibility of agents, enabling automatic quality control throughout synthesis. Extensive experiments demonstrate that models fine-tuned on CodeEvo data significantly outperform established baselines across code generation benchmarks with various difficulties. In-depth analyses further provide insights from multiple perspectives into effective code-centric data synthesis. 

---
# CIMR: Contextualized Iterative Multimodal Reasoning for Robust Instruction Following in LVLMs 

**Authors**: Yangshu Yuan, Heng Chen, Xinyi Jiang, Christian Ng, Kexin Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22074)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) has enhanced our ability to process and generate human language and visual information. However, these models often struggle with complex, multi-step multi-modal instructions that require logical reasoning, dynamic feedback integration, and iterative self-correction. To address this, we propose CIMR: Contextualized Iterative Multimodal Reasoning, a novel framework that introduces a context-aware iterative reasoning and self-correction module. CIMR operates in two stages: initial reasoning and response generation, followed by iterative refinement using parsed multi-modal feedback. A dynamic fusion module deeply integrates textual, visual, and contextual features at each step. We fine-tune LLaVA-1.5-7B on the Visual Instruction Tuning (VIT) dataset and evaluate CIMR on the newly introduced Multi-modal Action Planning (MAP) dataset. CIMR achieves 91.5% accuracy, outperforming state-of-the-art models such as GPT-4V (89.2%), LLaVA-1.5 (78.5%), MiniGPT-4 (75.3%), and InstructBLIP (72.8%), demonstrating the efficacy of its iterative reasoning and self-correction capabilities in complex tasks. 

---
