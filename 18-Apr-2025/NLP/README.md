# CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training 

**Authors**: Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Yingyan, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2504.13161)  

**Abstract**: Pre-training datasets are typically collected from web content and lack inherent domain divisions. For instance, widely used datasets like Common Crawl do not include explicit domain labels, while manually curating labeled datasets such as The Pile is labor-intensive. Consequently, identifying an optimal pre-training data mixture remains a challenging problem, despite its significant benefits for pre-training performance. To address these challenges, we propose CLustering-based Iterative Data Mixture Bootstrapping (CLIMB), an automated framework that discovers, evaluates, and refines data mixtures in a pre-training setting. Specifically, CLIMB embeds and clusters large-scale datasets in a semantic space and then iteratively searches for optimal mixtures using a smaller proxy model and a predictor. When continuously trained on 400B tokens with this mixture, our 1B model exceeds the state-of-the-art Llama-3.2-1B by 2.0%. Moreover, we observe that optimizing for a specific domain (e.g., Social Sciences) yields a 5% improvement over random sampling. Finally, we introduce ClimbLab, a filtered 1.2-trillion-token corpus with 20 clusters as a research playground, and ClimbMix, a compact yet powerful 400-billion-token dataset designed for efficient pre-training that delivers superior performance under an equal token budget. We analyze the final data mixture, elucidating the characteristics of an optimal data mixture. Our data is available at: this https URL 

---
# Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo 

**Authors**: João Loula, Benjamin LeBrun, Li Du, Ben Lipkin, Clemente Pasti, Gabriel Grand, Tianyu Liu, Yahya Emara, Marjorie Freedman, Jason Eisner, Ryan Cotterel, Vikash Mansinghka, Alexander K. Lew, Tim Vieira, Timothy J. O'Donnell  

**Link**: [PDF](https://arxiv.org/pdf/2504.13139)  

**Abstract**: A wide range of LM applications require generating text that conforms to syntactic or semantic constraints. Imposing such constraints can be naturally framed as probabilistic conditioning, but exact generation from the resulting distribution -- which can differ substantially from the LM's base distribution -- is generally intractable. In this work, we develop an architecture for controlled LM generation based on sequential Monte Carlo (SMC). Our SMC framework allows us to flexibly incorporate domain- and problem-specific constraints at inference time, and efficiently reallocate computational resources in light of new information during the course of generation. By comparing to a number of alternatives and ablations on four challenging domains -- Python code generation for data science, text-to-SQL, goal inference, and molecule synthesis -- we demonstrate that, with little overhead, our approach allows small open-source language models to outperform models over 8x larger, as well as closed-source, fine-tuned ones. In support of the probabilistic perspective, we show that these performance improvements are driven by better approximation to the posterior distribution. Our system builds on the framework of Lew et al. (2023) and integrates with its language model probabilistic programming language, giving users a simple, programmable way to apply SMC to a broad variety of controlled generation problems. 

---
# Energy-Based Reward Models for Robust Language Model Alignment 

**Authors**: Anamika Lochab, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13134)  

**Abstract**: Reward models (RMs) are essential for aligning Large Language Models (LLMs) with human preferences. However, they often struggle with capturing complex human preferences and generalizing to unseen data. To address these challenges, we introduce Energy-Based Reward Model (EBRM), a lightweight post-hoc refinement framework that enhances RM robustness and generalization. EBRM models the reward distribution explicitly, capturing uncertainty in human preferences and mitigating the impact of noisy or misaligned annotations. It achieves this through conflict-aware data filtering, label-noise-aware contrastive training, and hybrid initialization. Notably, EBRM enhances RMs without retraining, making it computationally efficient and adaptable across different models and tasks. Empirical evaluations on RM benchmarks demonstrate significant improvements in both robustness and generalization, achieving up to a 5.97% improvement in safety-critical alignment tasks compared to standard RMs. Furthermore, reinforcement learning experiments confirm that our refined rewards enhance alignment quality, effectively delaying reward hacking. These results demonstrate our approach as a scalable and effective enhancement for existing RMs and alignment pipelines. The code is available at EBRM. 

---
# LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard 

**Authors**: Varun Rao, Youran Sun, Mahendra Kumar, Tejas Mutneja, Agastya Mukherjee, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13125)  

**Abstract**: This paper investigates the application of large language models (LLMs) to financial tasks. We fine-tuned foundation models using the Open FinLLM Leaderboard as a benchmark. Building on Qwen2.5 and Deepseek-R1, we employed techniques including supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) to enhance their financial capabilities. The fine-tuned models demonstrated substantial performance gains across a wide range of financial tasks. Moreover, we measured the data scaling law in the financial domain. Our work demonstrates the potential of large language models (LLMs) in financial applications. 

---
# Retrieval-Augmented Generation with Conflicting Evidence 

**Authors**: Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.13079)  

**Abstract**: Large language model (LLM) agents are increasingly employing retrieval-augmented generation (RAG) to improve the factuality of their responses. However, in practice, these systems often need to handle ambiguous user queries and potentially conflicting information from multiple sources while also suppressing inaccurate information from noisy or irrelevant documents. Prior work has generally studied and addressed these challenges in isolation, considering only one aspect at a time, such as handling ambiguity or robustness to noise and misinformation. We instead consider multiple factors simultaneously, proposing (i) RAMDocs (Retrieval with Ambiguity and Misinformation in Documents), a new dataset that simulates complex and realistic scenarios for conflicting evidence for a user query, including ambiguity, misinformation, and noise; and (ii) MADAM-RAG, a multi-agent approach in which LLM agents debate over the merits of an answer over multiple rounds, allowing an aggregator to collate responses corresponding to disambiguated entities while discarding misinformation and noise, thereby handling diverse sources of conflict jointly. We demonstrate the effectiveness of MADAM-RAG using both closed and open-source models on AmbigDocs -- which requires presenting all valid answers for ambiguous queries -- improving over strong RAG baselines by up to 11.40% and on FaithEval -- which requires suppressing misinformation -- where we improve by up to 15.80% (absolute) with Llama3.3-70B-Instruct. Furthermore, we find that RAMDocs poses a challenge for existing RAG baselines (Llama3.3-70B-Instruct only obtains 32.60 exact match score). While MADAM-RAG begins to address these conflicting factors, our analysis indicates that a substantial gap remains especially when increasing the level of imbalance in supporting evidence and misinformation. 

---
# Accuracy is Not Agreement: Expert-Aligned Evaluation of Crash Narrative Classification Models 

**Authors**: Sudesh Ramesh Bhagat, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.13068)  

**Abstract**: This study explores the relationship between deep learning (DL) model accuracy and expert agreement in the classification of crash narratives. We evaluate five DL models -- including BERT variants, the Universal Sentence Encoder (USE), and a zero-shot classifier -- against expert-labeled data and narrative text. The analysis is further extended to four large language models (LLMs): GPT-4, LLaMA 3, Qwen, and Claude. Our results reveal a counterintuitive trend: models with higher technical accuracy often exhibit lower agreement with domain experts, whereas LLMs demonstrate greater expert alignment despite relatively lower accuracy scores. To quantify and interpret model-expert agreement, we employ Cohen's Kappa, Principal Component Analysis (PCA), and SHAP-based explainability techniques. Findings indicate that expert-aligned models tend to rely more on contextual and temporal language cues, rather than location-specific keywords. These results underscore that accuracy alone is insufficient for evaluating models in safety-critical NLP applications. We advocate for incorporating expert agreement as a complementary metric in model evaluation frameworks and highlight the promise of LLMs as interpretable, scalable tools for crash analysis pipelines. 

---
# Aspect-Based Summarization with Self-Aspect Retrieval Enhanced Generation 

**Authors**: Yichao Feng, Shuai Zhao, Yueqiu Li, Luwei Xiao, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13054)  

**Abstract**: Aspect-based summarization aims to generate summaries tailored to specific aspects, addressing the resource constraints and limited generalizability of traditional summarization approaches. Recently, large language models have shown promise in this task without the need for training. However, they rely excessively on prompt engineering and face token limits and hallucination challenges, especially with in-context learning. To address these challenges, in this paper, we propose a novel framework for aspect-based summarization: Self-Aspect Retrieval Enhanced Summary Generation. Rather than relying solely on in-context learning, given an aspect, we employ an embedding-driven retrieval mechanism to identify its relevant text segments. This approach extracts the pertinent content while avoiding unnecessary details, thereby mitigating the challenge of token limits. Moreover, our framework optimizes token usage by deleting unrelated parts of the text and ensuring that the model generates output strictly based on the given aspect. With extensive experiments on benchmark datasets, we demonstrate that our framework not only achieves superior performance but also effectively mitigates the token limitation problem. 

---
# ChatEXAONEPath: An Expert-level Multimodal Large Language Model for Histopathology Using Whole Slide Images 

**Authors**: Sangwook Kim, Soonyoung Lee, Jongseong Jang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13023)  

**Abstract**: Recent studies have made significant progress in developing large language models (LLMs) in the medical domain, which can answer expert-level questions and demonstrate the potential to assist clinicians in real-world clinical scenarios. Studies have also witnessed the importance of integrating various modalities with the existing LLMs for a better understanding of complex clinical contexts, which are innately multi-faceted by nature. Although studies have demonstrated the ability of multimodal LLMs in histopathology to answer questions from given images, they lack in understanding of thorough clinical context due to the patch-level data with limited information from public datasets. Thus, developing WSI-level MLLMs is significant in terms of the scalability and applicability of MLLMs in histopathology. In this study, we introduce an expert-level MLLM for histopathology using WSIs, dubbed as ChatEXAONEPath. We present a retrieval-based data generation pipeline using 10,094 pairs of WSIs and histopathology reports from The Cancer Genome Atlas (TCGA). We also showcase an AI-based evaluation protocol for a comprehensive understanding of the medical context from given multimodal information and evaluate generated answers compared to the original histopathology reports. We demonstrate the ability of diagnosing the given histopathology images using ChatEXAONEPath with the acceptance rate of 62.9% from 1,134 pairs of WSIs and reports. Our proposed model can understand pan-cancer WSIs and clinical context from various cancer types. We argue that our proposed model has the potential to assist clinicians by comprehensively understanding complex morphology of WSIs for cancer diagnosis through the integration of multiple modalities. 

---
# SHA256 at SemEval-2025 Task 4: Selective Amnesia -- Constrained Unlearning for Large Language Models via Knowledge Isolation 

**Authors**: Saransh Agrawal, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12996)  

**Abstract**: Large language models (LLMs) frequently memorize sensitive information during training, posing risks when deploying publicly accessible models. Current machine unlearning methods struggle to selectively remove specific data associations without degrading overall model capabilities. This paper presents our solution to SemEval-2025 Task 4 on targeted unlearning, which introduces a two-stage methodology that combines causal mediation analysis with layer-specific optimization. Through systematic causal tracing experiments on OLMo architectures (1B and 7B parameters), we identify the critical role of the first few transformer layers (layers 0-5) in storing subject-attribute associations within MLP modules. Building on this insight, we develop a constrained optimization approach that freezes upper layers while applying a novel joint loss function to lower layers-simultaneously maximizing forget set loss via output token cross-entropy penalties and minimizing retain set deviation through adaptive regularization. Our method achieves 2nd place in the 1B model track, demonstrating strong task performance while maintaining 88% of baseline MMLU accuracy. These results establish causal-informed layer optimization as a promising paradigm for efficient, precise unlearning in LLMs, offering a significant step forward in addressing data privacy concerns in AI systems. 

---
# Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild 

**Authors**: Jiatai Wang, Zhiwei Xu, Di Jin, Xuewen Yang, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12982)  

**Abstract**: The proliferation of large language models (LLMs) has significantly advanced information retrieval systems, particularly in response generation (RG). Unfortunately, LLMs often face knowledge conflicts between internal memory and retrievaled external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences. However, when the distinction is ambiguous, LLMs experience heightened uncertainty. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models into adaptive augmentation of retrieved information and guiding LLM preference in response generation. Extensive experiments on single-choice, open-ended question-answering (QA), and retrieval augmented generation (RAG) validate our theoretical findings and demonstrate the efficacy of Swin-VIB. Notably, our method improves single-choice task accuracy by at least 7.54\% over competitive baselines. 

---
# Sparks of Science: Hypothesis Generation Using Structured Paper Data 

**Authors**: Charles O'Neill, Tirthankar Ghosal, Roberta Răileanu, Mike Walmsley, Thang Bui, Kevin Schawinski, Ioana Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2504.12976)  

**Abstract**: Generating novel and creative scientific hypotheses is a cornerstone in achieving Artificial General Intelligence. Large language and reasoning models have the potential to aid in the systematic creation, selection, and validation of scientifically informed hypotheses. However, current foundation models often struggle to produce scientific ideas that are both novel and feasible. One reason is the lack of a dedicated dataset that frames Scientific Hypothesis Generation (SHG) as a Natural Language Generation (NLG) task. In this paper, we introduce HypoGen, the first dataset of approximately 5500 structured problem-hypothesis pairs extracted from top-tier computer science conferences structured with a Bit-Flip-Spark schema, where the Bit is the conventional assumption, the Spark is the key insight or conceptual leap, and the Flip is the resulting counterproposal. HypoGen uniquely integrates an explicit Chain-of-Reasoning component that reflects the intellectual process from Bit to Flip. We demonstrate that framing hypothesis generation as conditional language modelling, with the model fine-tuned on Bit-Flip-Spark and the Chain-of-Reasoning (and where, at inference, we only provide the Bit), leads to improvements in the overall quality of the hypotheses. Our evaluation employs automated metrics and LLM judge rankings for overall quality assessment. We show that by fine-tuning on our HypoGen dataset we improve the novelty, feasibility, and overall quality of the generated hypotheses. The HypoGen dataset is publicly available at this http URL. 

---
# Estimating Optimal Context Length for Hybrid Retrieval-augmented Multi-document Summarization 

**Authors**: Adithya Pratapa, Teruko Mitamura  

**Link**: [PDF](https://arxiv.org/pdf/2504.12972)  

**Abstract**: Recent advances in long-context reasoning abilities of language models led to interesting applications in large-scale multi-document summarization. However, prior work has shown that these long-context models are not effective at their claimed context windows. To this end, retrieval-augmented systems provide an efficient and effective alternative. However, their performance can be highly sensitive to the choice of retrieval context length. In this work, we present a hybrid method that combines retrieval-augmented systems with long-context windows supported by recent language models. Our method first estimates the optimal retrieval length as a function of the retriever, summarizer, and dataset. On a randomly sampled subset of the dataset, we use a panel of LLMs to generate a pool of silver references. We use these silver references to estimate the optimal context length for a given RAG system configuration. Our results on the multi-document summarization task showcase the effectiveness of our method across model classes and sizes. We compare against length estimates from strong long-context benchmarks such as RULER and HELMET. Our analysis also highlights the effectiveness of our estimation method for very long-context LMs and its generalization to new classes of LMs. 

---
# Are Retrials All You Need? Enhancing Large Language Model Reasoning Without Verbalized Feedback 

**Authors**: Nearchos Potamitis, Akhil Arora  

**Link**: [PDF](https://arxiv.org/pdf/2504.12951)  

**Abstract**: Recent advancements in large language models (LLMs) have catalyzed the development of general-purpose autonomous agents, demonstrating remarkable performance in complex reasoning tasks across various domains. This surge has spurred the evolution of a plethora of prompt-based reasoning frameworks. A recent focus has been on iterative reasoning strategies that refine outputs through self-evaluation and verbalized feedback. However, these strategies require additional computational complexity to enable models to recognize and correct their mistakes, leading to a significant increase in their cost. In this work, we introduce the concept of ``retrials without feedback'', an embarrassingly simple yet powerful mechanism for enhancing reasoning frameworks by allowing LLMs to retry problem-solving attempts upon identifying incorrect answers. Unlike conventional iterative refinement methods, our method does not require explicit self-reflection or verbalized feedback, simplifying the refinement process. Our findings indicate that simpler retrial-based approaches often outperform more sophisticated reasoning frameworks, suggesting that the benefits of complex methods may not always justify their computational costs. By challenging the prevailing assumption that more intricate reasoning strategies inherently lead to better performance, our work offers new insights into how simpler, more efficient approaches can achieve optimal results. So, are retrials all you need? 

---
# ConExion: Concept Extraction with Large Language Models 

**Authors**: Ebrahim Norouzi, Sven Hertling, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2504.12915)  

**Abstract**: In this paper, an approach for concept extraction from documents using pre-trained large language models (LLMs) is presented. Compared with conventional methods that extract keyphrases summarizing the important information discussed in a document, our approach tackles a more challenging task of extracting all present concepts related to the specific domain, not just the important ones. Through comprehensive evaluations of two widely used benchmark datasets, we demonstrate that our method improves the F1 score compared to state-of-the-art techniques. Additionally, we explore the potential of using prompts within these models for unsupervised concept extraction. The extracted concepts are intended to support domain coverage evaluation of ontologies and facilitate ontology learning, highlighting the effectiveness of LLMs in concept extraction tasks. Our source code and datasets are publicly available at this https URL. 

---
# MAIN: Mutual Alignment Is Necessary for instruction tuning 

**Authors**: Fanyi Yang, Jianfeng Liu, Xin Zhang, Haoyu Liu, Xixin Cao, Yuefeng Zhan, Hao Sun, Weiwei Deng, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12913)  

**Abstract**: Instruction tuning has enabled large language models (LLMs) to achieve remarkable performance, but its success heavily depends on the availability of large-scale, high-quality instruction-response pairs. However, current methods for scaling up data generation often overlook a crucial aspect: the alignment between instructions and responses. We hypothesize that high-quality instruction-response pairs are not defined by the individual quality of each component, but by the extent of their alignment with each other. To address this, we propose a Mutual Alignment Framework (MAIN) that ensures coherence between the instruction and response through mutual constraints. Experiments demonstrate that models such as LLaMA and Mistral, fine-tuned within this framework, outperform traditional methods across multiple benchmarks. This approach underscores the critical role of instruction-response alignment in enabling scalable and high-quality instruction tuning for LLMs. 

---
# Benchmarking Multi-National Value Alignment for Large Language Models 

**Authors**: Chengyi Ju, Weijie Shi, Chengzhong Liu, Jiaming Ji, Jipeng Zhang, Ruiyuan Zhang, Jia Zhu, Jiajie Xu, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.12911)  

**Abstract**: Do Large Language Models (LLMs) hold positions that conflict with your country's values? Occasionally they do! However, existing works primarily focus on ethical reviews, failing to capture the diversity of national values, which encompass broader policy, legal, and moral considerations. Furthermore, current benchmarks that rely on spectrum tests using manually designed questionnaires are not easily scalable.
To address these limitations, we introduce NaVAB, a comprehensive benchmark to evaluate the alignment of LLMs with the values of five major nations: China, the United States, the United Kingdom, France, and Germany. NaVAB implements a national value extraction pipeline to efficiently construct value assessment datasets. Specifically, we propose a modeling procedure with instruction tagging to process raw data sources, a screening process to filter value-related topics and a generation process with a Conflict Reduction mechanism to filter non-conflicting this http URL conduct extensive experiments on various LLMs across countries, and the results provide insights into assisting in the identification of misaligned scenarios. Moreover, we demonstrate that NaVAB can be combined with alignment techniques to effectively reduce value concerns by aligning LLMs' values with the target country. 

---
# Information Gain-Guided Causal Intervention for Autonomous Debiasing Large Language Models 

**Authors**: Zhouhao Sun, Xiao Ding, Li Du, Yunpeng Xu, Yixuan Ma, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12898)  

**Abstract**: Despite significant progress, recent studies indicate that current large language models (LLMs) may still capture dataset biases and utilize them during inference, leading to the poor generalizability of LLMs. However, due to the diversity of dataset biases and the insufficient nature of bias suppression based on in-context learning, the effectiveness of previous prior knowledge-based debiasing methods and in-context learning based automatic debiasing methods is limited. To address these challenges, we explore the combination of causal mechanisms with information theory and propose an information gain-guided causal intervention debiasing (IGCIDB) framework. This framework first utilizes an information gain-guided causal intervention method to automatically and autonomously balance the distribution of instruction-tuning dataset. Subsequently, it employs a standard supervised fine-tuning process to train LLMs on the debiased dataset. Experimental results show that IGCIDB can effectively debias LLM to improve its generalizability across different tasks. 

---
# Are AI agents the new machine translation frontier? Challenges and opportunities of single- and multi-agent systems for multilingual digital communication 

**Authors**: Vicent Briva-Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2504.12891)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has introduced AI agents as a disruptive paradigm across various industries, yet their application in machine translation (MT) remains underexplored. This paper describes and analyses the potential of single- and multi-agent systems for MT, reflecting on how they could enhance multilingual digital communication. While single-agent systems are well-suited for simpler translation tasks, multi-agent systems, which involve multiple specialized AI agents collaborating in a structured manner, may offer a promising solution for complex scenarios requiring high accuracy, domain-specific knowledge, and contextual awareness. To demonstrate the feasibility of multi-agent workflows in MT, we are conducting a pilot study in legal MT. The study employs a multi-agent system involving four specialized AI agents for (i) translation, (ii) adequacy review, (iii) fluency review, and (iv) final editing. Our findings suggest that multi-agent systems may have the potential to significantly improve domain-adaptability and contextual awareness, with superior translation quality to traditional MT or single-agent systems. This paper also sets the stage for future research into multi-agent applications in MT, integration into professional translation workflows, and shares a demo of the system analyzed in the paper. 

---
# ViClaim: A Multilingual Multilabel Dataset for Automatic Claim Detection in Videos 

**Authors**: Patrick Giedemann, Pius von Däniken, Jan Deriu, Alvaro Rodrigo, Anselmo Peñas, Mark Cieliebak  

**Link**: [PDF](https://arxiv.org/pdf/2504.12882)  

**Abstract**: The growing influence of video content as a medium for communication and misinformation underscores the urgent need for effective tools to analyze claims in multilingual and multi-topic settings. Existing efforts in misinformation detection largely focus on written text, leaving a significant gap in addressing the complexity of spoken text in video transcripts. We introduce ViClaim, a dataset of 1,798 annotated video transcripts across three languages (English, German, Spanish) and six topics. Each sentence in the transcripts is labeled with three claim-related categories: fact-check-worthy, fact-non-check-worthy, or opinion. We developed a custom annotation tool to facilitate the highly complex annotation process. Experiments with state-of-the-art multilingual language models demonstrate strong performance in cross-validation (macro F1 up to 0.896) but reveal challenges in generalization to unseen topics, particularly for distinct domains. Our findings highlight the complexity of claim detection in video transcripts. ViClaim offers a robust foundation for advancing misinformation detection in video-based communication, addressing a critical gap in multimodal analysis. 

---
# Can LLMs reason over extended multilingual contexts? Towards long-context evaluation beyond retrieval and haystacks 

**Authors**: Amey Hengle, Prasoon Bajpai, Soham Dan, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2504.12845)  

**Abstract**: Existing multilingual long-context benchmarks, often based on the popular needle-in-a-haystack test, primarily evaluate a model's ability to locate specific information buried within irrelevant texts. However, such a retrieval-centric approach is myopic and inherently limited, as successful recall alone does not indicate a model's capacity to reason over extended contexts. Moreover, these benchmarks are susceptible to data leakage, short-circuiting, and risk making the evaluation a priori identifiable. To address these limitations, we introduce MLRBench, a new synthetic benchmark for multilingual long-context reasoning. Unlike existing benchmarks, MLRBench goes beyond surface-level retrieval by including tasks that assess multi-hop inference, aggregation, and epistemic reasoning. Spanning seven languages, MLRBench is designed to be parallel, resistant to leakage, and scalable to arbitrary context lengths. Our extensive experiments with an open-weight large language model (LLM) reveal a pronounced gap between high- and low-resource languages, particularly for tasks requiring the model to aggregate multiple facts or predict the absence of information. We also find that, in multilingual settings, LLMs effectively utilize less than 30% of their claimed context length. Although off-the-shelf Retrieval Augmented Generation helps alleviate this to a certain extent, it does not solve the long-context problem. We open-source MLRBench to enable future research in improved evaluation and training of multilingual LLMs. 

---
# SMARTe: Slot-based Method for Accountable Relational Triple extraction 

**Authors**: Xue Wen Tan, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2504.12816)  

**Abstract**: Relational Triple Extraction (RTE) is a fundamental task in Natural Language Processing (NLP). However, prior research has primarily focused on optimizing model performance, with limited efforts to understand the internal mechanisms driving these models. Many existing methods rely on complex preprocessing to induce specific interactions, often resulting in opaque systems that may not fully align with their theoretical foundations. To address these limitations, we propose SMARTe: a Slot-based Method for Accountable Relational Triple extraction. SMARTe introduces intrinsic interpretability through a slot attention mechanism and frames the task as a set prediction problem. Slot attention consolidates relevant information into distinct slots, ensuring all predictions can be explicitly traced to learned slot representations and the tokens contributing to each predicted relational triple. While emphasizing interpretability, SMARTe achieves performance comparable to state-of-the-art models. Evaluations on the NYT and WebNLG datasets demonstrate that adding interpretability does not compromise performance. Furthermore, we conducted qualitative assessments to showcase the explanations provided by SMARTe, using attention heatmaps that map to their respective tokens. We conclude with a discussion of our findings and propose directions for future research. 

---
# Assesing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation 

**Authors**: Takaya Arita, Wenxian Zheng, Reiji Suzuki, Fuminori Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2504.12805)  

**Abstract**: This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume. 

---
# Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration 

**Authors**: Yicheng Pan, Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Jun Du, Jianshu Zhang, Quan Liu, Jianqing Gao, Feng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12773)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have achieved remarkable progress in general domains and demonstrated promise in multimodal mathematical reasoning. However, applying MLLMs to geometry problem solving (GPS) remains challenging due to lack of accurate step-by-step solution data and severe hallucinations during reasoning. In this paper, we propose GeoGen, a pipeline that can automatically generates step-wise reasoning paths for geometry diagrams. By leveraging the precise symbolic reasoning, \textbf{GeoGen} produces large-scale, high-quality question-answer pairs. To further enhance the logical reasoning ability of MLLMs, we train \textbf{GeoLogic}, a Large Language Model (LLM) using synthetic data generated by GeoGen. Serving as a bridge between natural language and symbolic systems, GeoLogic enables symbolic tools to help verifying MLLM outputs, making the reasoning process more rigorous and alleviating hallucinations. Experimental results show that our approach consistently improves the performance of MLLMs, achieving remarkable results on benchmarks for geometric reasoning tasks. This improvement stems from our integration of the strengths of LLMs and symbolic systems, which enables a more reliable and interpretable approach for the GPS task. Codes are available at this https URL. 

---
# Out of Sight Out of Mind, Out of Sight Out of Mind: Measuring Bias in Language Models Against Overlooked Marginalized Groups in Regional Contexts 

**Authors**: Fatma Elsafoury, David Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.12767)  

**Abstract**: We know that language models (LMs) form biases and stereotypes of minorities, leading to unfair treatments of members of these groups, thanks to research mainly in the US and the broader English-speaking world. As the negative behavior of these models has severe consequences for society and individuals, industry and academia are actively developing methods to reduce the bias in LMs. However, there are many under-represented groups and languages that have been overlooked so far. This includes marginalized groups that are specific to individual countries and regions in the English speaking and Western world, but crucially also almost all marginalized groups in the rest of the world. The UN estimates, that between 600 million to 1.2 billion people worldwide are members of marginalized groups and in need for special protection. If we want to develop inclusive LMs that work for everyone, we have to broaden our understanding to include overlooked marginalized groups and low-resource languages and dialects.
In this work, we contribute to this effort with the first study investigating offensive stereotyping bias in 23 LMs for 270 marginalized groups from Egypt, the remaining 21 Arab countries, Germany, the UK, and the US. Additionally, we investigate the impact of low-resource languages and dialects on the study of bias in LMs, demonstrating the limitations of current bias metrics, as we measure significantly higher bias when using the Egyptian Arabic dialect versus Modern Standard Arabic. Our results show, LMs indeed show higher bias against many marginalized groups in comparison to dominant groups. However, this is not the case for Arabic LMs, where the bias is high against both marginalized and dominant groups in relation to religion and ethnicity.
Our results also show higher intersectional bias against Non-binary, LGBTQIA+ and Black women. 

---
# Chinese-Vicuna: A Chinese Instruction-following Llama-based Model 

**Authors**: Chenghao Fan, Zhenyi Lu, Jie Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.12737)  

**Abstract**: Chinese-Vicuna is an open-source, resource-efficient language model designed to bridge the gap in Chinese instruction-following capabilities by fine-tuning Meta's LLaMA architecture using Low-Rank Adaptation (LoRA). Targeting low-resource environments, it enables cost-effective deployment on consumer GPUs (e.g., RTX-2080Ti for 7B models) and supports domain-specific adaptation in fields like healthcare and law. By integrating hybrid datasets (BELLE and Guanaco) and 4-bit quantization (QLoRA), the model achieves competitive performance in tasks such as translation, code generation, and domain-specific Q\&A. The project provides a comprehensive toolkit for model conversion, CPU inference, and multi-turn dialogue interfaces, emphasizing accessibility for researchers and developers. Evaluations indicate competitive performance across medical tasks, multi-turn dialogue coherence, and real-time legal updates. Chinese-Vicuna's modular design, open-source ecosystem, and community-driven enhancements position it as a versatile foundation for Chinese LLM applications. 

---
# Pandora: A Code-Driven Large Language Model Agent for Unified Reasoning Across Diverse Structured Knowledge 

**Authors**: Yongrui Chen, Junhao He, Linbo Fu, Shenyu Zhang, Rihui Jin, Xinbang Dai, Jiaqi Li, Dehai Min, Nan Hu, Yuxin Zhang, Guilin Qi, Yi Huang, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12734)  

**Abstract**: Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions (NLQs) by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods either rely on employing task-specific strategies or custom-defined representations, which struggle to leverage the knowledge transfer between different SKR tasks or align with the prior of LLMs, thereby limiting their performance. This paper proposes a novel USKR framework named \textsc{Pandora}, which takes advantage of \textsc{Python}'s \textsc{Pandas} API to construct a unified knowledge representation for alignment with LLM pre-training. It employs an LLM to generate textual reasoning steps and executable Python code for each question. Demonstrations are drawn from a memory of training examples that cover various SKR tasks, facilitating knowledge transfer. Extensive experiments on four benchmarks involving three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified frameworks and competes effectively with task-specific methods. 

---
# KODIS: A Multicultural Dispute Resolution Dialogue Corpus 

**Authors**: James Hale, Sushrita Rakshit, Kushal Chawla, Jeanne M. Brett, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2504.12723)  

**Abstract**: We present KODIS, a dyadic dispute resolution corpus containing thousands of dialogues from over 75 countries. Motivated by a theoretical model of culture and conflict, participants engage in a typical customer service dispute designed by experts to evoke strong emotions and conflict. The corpus contains a rich set of dispositional, process, and outcome measures. The initial analysis supports theories of how anger expressions lead to escalatory spirals and highlights cultural differences in emotional expression. We make this corpus and data collection framework available to the community. 

---
# Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations 

**Authors**: Yiyou Sun, Yu Gai, Lijie Chen, Abhilasha Ravichander, Yejin Choi, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.12691)  

**Abstract**: Large language models (LLMs) frequently generate hallucinations-content that deviates from factual accuracy or provided context-posing challenges for diagnosis due to the complex interplay of underlying causes. This paper introduces a subsequence association framework to systematically trace and understand hallucinations. Our key insight is that hallucinations arise when dominant hallucinatory associations outweigh faithful ones. Through theoretical and empirical analyses, we demonstrate that decoder-only transformers effectively function as subsequence embedding models, with linear layers encoding input-output associations. We propose a tracing algorithm that identifies causal subsequences by analyzing hallucination probabilities across randomized input contexts. Experiments show our method outperforms standard attribution techniques in identifying hallucination causes and aligns with evidence from the model's training corpus. This work provides a unified perspective on hallucinations and a robust framework for their tracing and analysis. 

---
# Data-efficient LLM Fine-tuning for Code Generation 

**Authors**: Weijie Lv, Xuan Xia, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12687)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in code generation tasks. However, there remains a performance gap between open-source and closed-source models. To address this gap, existing approaches typically generate large amounts of synthetic data for fine-tuning, which often leads to inefficient training. In this work, we propose a data selection strategy in order to improve the effectiveness and efficiency of training for code-based LLMs. By prioritizing data complexity and ensuring that the sampled subset aligns with the distribution of the original dataset, our sampling strategy effectively selects high-quality data. Additionally, we optimize the tokenization process through a "dynamic pack" technique, which minimizes padding tokens and reduces computational resource consumption. Experimental results show that when training on 40% of the OSS-Instruct dataset, the DeepSeek-Coder-Base-6.7B model achieves an average performance of 66.9%, surpassing the 66.1% performance with the full dataset. Moreover, training time is reduced from 47 minutes to 34 minutes, and the peak GPU memory decreases from 61.47 GB to 42.72 GB during a single epoch. Similar improvements are observed with the CodeLlama-Python-7B model on the Evol-Instruct dataset. By optimizing both data selection and tokenization, our approach not only improves model performance but also improves training efficiency. 

---
# GRAIL: Gradient-Based Adaptive Unlearning for Privacy and Copyright in LLMs 

**Authors**: Kun-Woo Kim, Ji-Hoon Park, Ju-Min Han, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.12681)  

**Abstract**: Large Language Models (LLMs) trained on extensive datasets often learn sensitive information, which raises significant social and legal concerns under principles such as the "Right to be forgotten." Retraining entire models from scratch to remove undesired information is both costly and impractical. Furthermore, existing single-domain unlearning methods fail to address multi-domain scenarios, where knowledge is interwoven across domains such as privacy and copyright, creating overlapping representations that lead to excessive knowledge removal or degraded performance. To tackle these issues, we propose GRAIL (GRadient-based AdaptIve unLearning), a novel multi-domain unlearning framework. GRAIL leverages gradient information from multiple domains to precisely distinguish the unlearning scope from the retention scope, and applies an adaptive parameter-wise localization strategy to selectively remove targeted knowledge while preserving critical parameters for each domain. Experimental results on unlearning benchmarks show that GRAIL achieves unlearning success on par with the existing approaches, while also demonstrating up to 17% stronger knowledge retention success compared to the previous state-of-art method. Our findings establish a new paradigm for effectively managing and regulating sensitive information in large-scale pre-trained language models. 

---
# ACoRN: Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models 

**Authors**: Singon Kim, Gunho Jung, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.12673)  

**Abstract**: Abstractive compression utilizes smaller langauge models to condense query-relevant context, reducing computational costs in retrieval-augmented generation (RAG). However,retrieved documents often include information that is either irrelevant to answering the query or misleading due to factual incorrect content, despite having high relevance scores. This behavior indicates that abstractive compressors are more likely to omit important information essential for the correct answer, especially in long contexts where attention dispersion occurs. To address this issue, we categorize retrieved documents in a more fine-grained manner and propose Abstractive Compression Robust against Noise (ACoRN), which introduces two novel training steps. First, we use offline data augmentation on the training dataset to enhance compressor robustness against two distinct types of retrieval noise. Second, since the language modelbased compressor cannot fully utilize information from multiple retrieved documents and exhibits positional bias, we perform finetuning to generate summaries centered around key information that directly supports the correct answer. Our experiments demonstrate that T5-large, trained with ACoRN as a compressor, improves EM and F1 scores while preserving the answer string, which could serve as direct evidence. ACoRN excels on datasets with many accuracy-reducing documents, making it highly useful in real-world scenarios. 

---
# Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment 

**Authors**: Xiaotian Zhang, Ruizhe Chen, Yang Feng, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12663)  

**Abstract**: Aligning language models with human preferences presents significant challenges, particularly in achieving personalization without incurring excessive computational costs. Existing methods rely on reward signals and additional annotated data, limiting their scalability and adaptability to diverse human values. To address these challenges, we introduce Persona-judge, a novel discriminative paradigm that enables training-free personalized alignment with unseen preferences. Instead of optimizing policy parameters through external reward feedback, Persona-judge leverages the intrinsic preference judgment capabilities of the model. Specifically, a draft model generates candidate tokens conditioned on a given preference, while a judge model, embodying another preference, cross-validates the predicted tokens whether to be accepted. Experimental results demonstrate that Persona-judge, using the inherent preference evaluation mechanisms of the model, offers a scalable and computationally efficient solution to personalized alignment, paving the way for more adaptive customized alignment. 

---
# Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation 

**Authors**: Linda He, Jue Wang, Maurice Weber, Shang Zhu, Ben Athiwaratkun, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12637)  

**Abstract**: Large Language Models (LLMs) struggle with long-context reasoning, not only due to the quadratic scaling of computational complexity with sequence length but also because of the scarcity and expense of annotating long-context data. There has been barely any open-source work that systematically ablates long-context data, nor is there any openly available instruction tuning dataset with contexts surpassing 100K tokens. To bridge this gap, we introduce a novel post-training synthetic data generation strategy designed to efficiently extend the context window of LLMs while preserving their general task performance. Our approach scalably extends to arbitrarily long context lengths, unconstrained by the length of available real-world data, which effectively addresses the scarcity of raw long-context data. Through a step-by-step rotary position embedding (RoPE) scaling training strategy, we demonstrate that our model, with a context length of up to 1M tokens, performs well on the RULER benchmark and InfiniteBench and maintains robust performance on general language tasks. 

---
# Towards Characterizing Subjectivity of Individuals through Modeling Value Conflicts and Trade-offs 

**Authors**: Younghun Lee, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.12633)  

**Abstract**: Large Language Models (LLMs) not only have solved complex reasoning problems but also exhibit remarkable performance in tasks that require subjective decision making. Existing studies suggest that LLM generations can be subjectively grounded to some extent, yet exploring whether LLMs can account for individual-level subjectivity has not been sufficiently studied. In this paper, we characterize subjectivity of individuals on social media and infer their moral judgments using LLMs. We propose a framework, SOLAR (Subjective Ground with Value Abstraction), that observes value conflicts and trade-offs in the user-generated texts to better represent subjective ground of individuals. Empirical results show that our framework improves overall inference results as well as performance on controversial situations. Additionally, we qualitatively show that SOLAR provides explanations about individuals' value preferences, which can further account for their judgments. 

---
# GeoSense: Evaluating Identification and Application of Geometric Principles in Multimodal Reasoning 

**Authors**: Liangyu Xu, Yingxiu Zhao, Jingyun Wang, Yingyao Wang, Bu Pi, Chen Wang, Mingliang Zhang, Jihao Gu, Xiang Li, Xiaoyong Zhu, Jun Song, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.12597)  

**Abstract**: Geometry problem-solving (GPS), a challenging task requiring both visual comprehension and symbolic reasoning, effectively measures the reasoning capabilities of multimodal large language models (MLLMs). Humans exhibit strong reasoning ability in this task through accurate identification and adaptive application of geometric principles within visual contexts. However, existing benchmarks fail to jointly assess both dimensions of the human-like geometric reasoning mechanism in MLLMs, remaining a critical gap in assessing their ability to tackle GPS. To this end, we introduce GeoSense, the first comprehensive bilingual benchmark designed to systematically evaluate the geometric reasoning abilities of MLLMs through the lens of geometric principles. GeoSense features a five-level hierarchical framework of geometric principles spanning plane and solid geometry, an intricately annotated dataset of 1,789 problems, and an innovative evaluation strategy. Through extensive experiments on GeoSense with various open-source and closed-source MLLMs, we observe that Gemini-2.0-pro-flash performs best, achieving an overall score of $65.3$. Our in-depth analysis reveals that the identification and application of geometric principles remain a bottleneck for leading MLLMs, jointly hindering their reasoning abilities. These findings underscore GeoSense's potential to guide future advancements in MLLMs' geometric reasoning capabilities, paving the way for more robust and human-like reasoning in artificial intelligence. 

---
# Identifying and Mitigating the Influence of the Prior Distribution in Large Language Models 

**Authors**: Liyi Zhang, Veniamin Veselovsky, R. Thomas McCoy, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2504.12585)  

**Abstract**: Large language models (LLMs) sometimes fail to respond appropriately to deterministic tasks -- such as counting or forming acronyms -- because the implicit prior distribution they have learned over sequences of tokens influences their responses. In this work, we show that, in at least some cases, LLMs actually compute the information needed to perform these tasks correctly, and we identify some interventions that can allow them to access this information to improve their performance. First, we show that simply prompting the language model to not rely on its prior knowledge leads to dramatic improvements in prior-dominated tasks. We then use mechanistic interpretability techniques to localize the prior within the LLM and manipulate the extent to which that prior influences its responses. Specifically, we show that it is possible to identify layers of the underlying neural network that correlate with the prior probability of a response and that lightweight finetuning of these layers with basic prompts on prior-dominated tasks achieves high performance on held-out answers. These results suggest that the information required to produce a correct response is contained within the representations of the problems formed by the models. Furthermore, we show that this finetuning is significantly more effective for prior-dominated tasks, and that the error after finetuning is no longer correlated with the prior. Our results suggest that it may be possible to define effective methods for manipulating the extent to which LLMs rely upon their priors in solving problems, potentially increasing their performance in settings where LLMs hallucinate for reasons related to the prior probability of token sequences. 

---
# MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation 

**Authors**: Haris Riaz, Sourav Bhabesh, Vinayak Arannil, Miguel Ballesteros, Graham Horwood  

**Link**: [PDF](https://arxiv.org/pdf/2504.12563)  

**Abstract**: Recent smaller language models such Phi-3.5 and Phi-4 rely on synthetic data generated using larger Language models. Questions remain about leveraging synthetic data for other use cases, such as adapting LLMs to specific domains. A key limitation of synthetic data is low diversity, which negatively impacts its downstream applicability for improving other models. To address this, we propose MetaSynth, a method for generating synthetic data that enhances diversity through meta-prompting, where a language model orchestrates multiple "expert" LLM agents to collaboratively generate data. Using only 25 million tokens of synthetic data generated with MetaSynth, we successfully adapt a well-trained LLM (Mistral-7B-v0.3) to two specialized domains-Finance and Biomedicine-without compromising the capabilities of the resulting model in general tasks. In addition, we evaluate the diversity of our synthetic data using seven automated metrics, and find that it approaches the diversity of LLM pre-training corpora.
Continually pre-training Mistral-7B-v0.3 with MetaSynth notably outperforms the base LLM, showing improvements of up to 4.08% in Finance and 13.75% in Biomedicine. The same model shows degraded performance when trained on data generated using a template prompt, even when the template includes prior generations and varying In-Context exemplars of real data. Our findings suggest that a few million tokens of diverse synthetic data without mixing any real data, is sufficient for effective domain adaptation when using MetaSynth. 

---
# CDF-RAG: Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation 

**Authors**: Elahe Khatibi, Ziyu Wang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12560)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced large language models (LLMs) in knowledge-intensive tasks by incorporating external knowledge retrieval. However, existing RAG frameworks primarily rely on semantic similarity and correlation-driven retrieval, limiting their ability to distinguish true causal relationships from spurious associations. This results in responses that may be factually grounded but fail to establish cause-and-effect mechanisms, leading to incomplete or misleading insights. To address this issue, we introduce Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation (CDF-RAG), a framework designed to improve causal consistency, factual accuracy, and explainability in generative reasoning. CDF-RAG iteratively refines queries, retrieves structured causal graphs, and enables multi-hop causal reasoning across interconnected knowledge sources. Additionally, it validates responses against causal pathways, ensuring logically coherent and factually grounded outputs. We evaluate CDF-RAG on four diverse datasets, demonstrating its ability to improve response accuracy and causal correctness over existing RAG-based methods. Our code is publicly available at this https URL elakhatibi/CDF-RAG. 

---
# ELAB: Extensive LLM Alignment Benchmark in Persian Language 

**Authors**: Zahra Pourbahman, Fatemeh Rajabi, Mohammadhossein Sadeghi, Omid Ghahroodi, Somaye Bakhshaei, Arash Amini, Reza Kazemi, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2504.12553)  

**Abstract**: This paper presents a comprehensive evaluation framework for aligning Persian Large Language Models (LLMs) with critical ethical dimensions, including safety, fairness, and social norms. It addresses the gaps in existing LLM evaluation frameworks by adapting them to Persian linguistic and cultural contexts. This benchmark creates three types of Persian-language benchmarks: (i) translated data, (ii) new data generated synthetically, and (iii) new naturally collected data. We translate Anthropic Red Teaming data, AdvBench, HarmBench, and DecodingTrust into Persian. Furthermore, we create ProhibiBench-fa, SafeBench-fa, FairBench-fa, and SocialBench-fa as new datasets to address harmful and prohibited content in indigenous culture. Moreover, we collect extensive dataset as GuardBench-fa to consider Persian cultural norms. By combining these datasets, our work establishes a unified framework for evaluating Persian LLMs, offering a new approach to culturally grounded alignment evaluation. A systematic evaluation of Persian LLMs is performed across the three alignment aspects: safety (avoiding harmful content), fairness (mitigating biases), and social norms (adhering to culturally accepted behaviors). We present a publicly available leaderboard that benchmarks Persian LLMs with respect to safety, fairness, and social norms at: this https URL. 

---
# Memorization: A Close Look at Books 

**Authors**: Iris Ma, Ian Domingo, Alberto Krone-Martins, Pierre Baldi, Cristina V. Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2504.12549)  

**Abstract**: To what extent can entire books be extracted from LLMs? Using the Llama 3 70B family of models, and the "prefix-prompting" extraction technique, we were able to auto-regressively reconstruct, with a very high level of similarity, one entire book (Alice's Adventures in Wonderland) from just the first 500 tokens. We were also able to obtain high extraction rates on several other books, piece-wise. However, these successes do not extend uniformly to all books. We show that extraction rates of books correlate with book popularity and thus, likely duplication in the training data.
We also confirm the undoing of mitigations in the instruction-tuned Llama 3.1, following recent work (Nasr et al., 2025). We further find that this undoing comes from changes to only a tiny fraction of weights concentrated primarily in the lower transformer blocks. Our results provide evidence of the limits of current regurgitation mitigation strategies and introduce a framework for studying how fine-tuning affects the retrieval of verbatim memorization in aligned LLMs. 

---
# Memorization vs. Reasoning: Updating LLMs with New Knowledge 

**Authors**: Aochong Oliver Li, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2504.12523)  

**Abstract**: Large language models (LLMs) encode vast amounts of pre-trained knowledge in their parameters, but updating them as real-world information evolves remains a challenge. Existing methodologies and benchmarks primarily target entity substitutions, failing to capture the full breadth of complex real-world dynamics. In this paper, we introduce Knowledge Update Playground (KUP), an automatic pipeline for simulating realistic knowledge updates reflected in an evidence corpora. KUP's evaluation framework includes direct and indirect probes to both test memorization of updated facts and reasoning over them, for any update learning methods. Next, we present a lightweight method called memory conditioned training (MCT), which conditions tokens in the update corpus on self-generated "memory" tokens during training. Our strategy encourages LLMs to surface and reason over newly memorized knowledge at inference. Our results on two strong LLMs show that (1) KUP benchmark is highly challenging, with the best CPT models achieving $<2\%$ in indirect probing setting (reasoning) and (2) MCT training significantly outperforms prior continued pre-training (CPT) baselines, improving direct probing (memorization) results by up to $25.4\%$. 

---
# Evaluating the Diversity and Quality of LLM Generated Content 

**Authors**: Alexander Shypula, Shuo Li, Botong Zhang, Vishakh Padmakumar, Kayo Yin, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12522)  

**Abstract**: Recent work suggests that preference-tuning techniques--including Reinforcement Learning from Human Preferences (RLHF) methods like PPO and GRPO, as well as alternatives like DPO--reduce diversity, creating a dilemma given that such models are widely deployed in applications requiring diverse outputs. To address this, we introduce a framework for measuring effective semantic diversity--diversity among outputs that meet quality thresholds--which better reflects the practical utility of large language models (LLMs). Using open-ended tasks that require no human intervention, we find counterintuitive results: although preference-tuned models--especially those trained via RL--exhibit reduced lexical and syntactic diversity, they produce greater effective semantic diversity than SFT or base models, not from increasing diversity among high-quality outputs, but from generating more high-quality outputs overall. We discover that preference tuning reduces syntactic diversity while preserving semantic diversity--revealing a distinction between diversity in form and diversity in content that traditional metrics often overlook. Our analysis further shows that smaller models are consistently more parameter-efficient at generating unique content within a fixed sampling budget, offering insights into the relationship between model scaling and diversity. These findings have important implications for applications that require diverse yet high-quality outputs, from creative assistance to synthetic data generation. 

---
# BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents 

**Authors**: Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, Amelia Glaese  

**Link**: [PDF](https://arxiv.org/pdf/2504.12516)  

**Abstract**: We present BrowseComp, a simple yet challenging benchmark for measuring the ability for agents to browse the web. BrowseComp comprises 1,266 questions that require persistently navigating the internet in search of hard-to-find, entangled information. Despite the difficulty of the questions, BrowseComp is simple and easy-to-use, as predicted answers are short and easily verifiable against reference answers. BrowseComp for browsing agents can be seen as analogous to how programming competitions are an incomplete but useful benchmark for coding agents. While BrowseComp sidesteps challenges of a true user query distribution, like generating long answers or resolving ambiguity, it measures the important core capability of exercising persistence and creativity in finding information. BrowseComp can be found at this https URL. 

---
# Beyond Text: Characterizing Domain Expert Needs in Document Research 

**Authors**: Sireesh Gururaja, Nupoor Gandhi, Jeremiah Milbauer, Emma Strubell  

**Link**: [PDF](https://arxiv.org/pdf/2504.12495)  

**Abstract**: Working with documents is a key part of almost any knowledge work, from contextualizing research in a literature review to reviewing legal precedent. Recently, as their capabilities have expanded, primarily text-based NLP systems have often been billed as able to assist or even automate this kind of work. But to what extent are these systems able to model these tasks as experts conceptualize and perform them now? In this study, we interview sixteen domain experts across two domains to understand their processes of document research, and compare it to the current state of NLP systems. We find that our participants processes are idiosyncratic, iterative, and rely extensively on the social context of a document in addition its content; existing approaches in NLP and adjacent fields that explicitly center the document as an object, rather than as merely a container for text, tend to better reflect our participants' priorities, though they are often less accessible outside their research communities. We call on the NLP community to more carefully consider the role of the document in building useful tools that are accessible, personalizable, iterative, and socially aware. 

---
# Accelerating Clinical NLP at Scale with a Hybrid Framework with Reduced GPU Demands: A Case Study in Dementia Identification 

**Authors**: Jianlin Shi, Qiwei Gan, Elizabeth Hanchrow, Annie Bowles, John Stanley, Adam P. Bress, Jordana B. Cohen, Patrick R. Alba  

**Link**: [PDF](https://arxiv.org/pdf/2504.12494)  

**Abstract**: Clinical natural language processing (NLP) is increasingly in demand in both clinical research and operational practice. However, most of the state-of-the-art solutions are transformers-based and require high computational resources, limiting their accessibility. We propose a hybrid NLP framework that integrates rule-based filtering, a Support Vector Machine (SVM) classifier, and a BERT-based model to improve efficiency while maintaining accuracy. We applied this framework in a dementia identification case study involving 4.9 million veterans with incident hypertension, analyzing 2.1 billion clinical notes. At the patient level, our method achieved a precision of 0.90, a recall of 0.84, and an F1-score of 0.87. Additionally, this NLP approach identified over three times as many dementia cases as structured data methods. All processing was completed in approximately two weeks using a single machine with dual A40 GPUs. This study demonstrates the feasibility of hybrid NLP solutions for large-scale clinical text analysis, making state-of-the-art methods more accessible to healthcare organizations with limited computational resources. 

---
# Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs? 

**Authors**: Hansi Zeng, Kai Hui, Honglei Zhuang, Zhen Qin, Zhenrui Yue, Hamed Zamani, Dana Alon  

**Link**: [PDF](https://arxiv.org/pdf/2504.12491)  

**Abstract**: While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks. 

---
# Integrating Structural and Semantic Signals in Text-Attributed Graphs with BiGTex 

**Authors**: Azadeh Beiranvand, Seyed Mehdi Vahidipour  

**Link**: [PDF](https://arxiv.org/pdf/2504.12474)  

**Abstract**: Text-attributed graphs (TAGs) present unique challenges in representation learning by requiring models to capture both the semantic richness of node-associated texts and the structural dependencies of the graph. While graph neural networks (GNNs) excel at modeling topological information, they lack the capacity to process unstructured text. Conversely, large language models (LLMs) are proficient in text understanding but are typically unaware of graph structure. In this work, we propose BiGTex (Bidirectional Graph Text), a novel architecture that tightly integrates GNNs and LLMs through stacked Graph-Text Fusion Units. Each unit allows for mutual attention between textual and structural representations, enabling information to flow in both directions, text influencing structure and structure guiding textual interpretation. The proposed architecture is trained using parameter-efficient fine-tuning (LoRA), keeping the LLM frozen while adapting to task-specific signals. Extensive experiments on five benchmark datasets demonstrate that BiGTex achieves state-of-the-art performance in node classification and generalizes effectively to link prediction. An ablation study further highlights the importance of soft prompting and bi-directional attention in the model's success. 

---
# SLURG: Investigating the Feasibility of Generating Synthetic Online Fallacious Discourse 

**Authors**: Cal Blanco, Gavin Dsouza, Hugo Lin, Chelsey Rush  

**Link**: [PDF](https://arxiv.org/pdf/2504.12466)  

**Abstract**: In our paper we explore the definition, and extrapolation of fallacies as they pertain to the automatic detection of manipulation on social media. In particular we explore how these logical fallacies might appear in the real world i.e internet forums. We discovered a prevalence of misinformation / misguided intention in discussion boards specifically centered around the Ukrainian Russian Conflict which serves to narrow the domain of our task. Although automatic fallacy detection has gained attention recently, most datasets use unregulated fallacy taxonomies or are limited to formal linguistic domains like political debates or news reports. Online discourse, however, often features non-standardized and diverse language not captured in these domains. We present Shady Linguistic Utterance Replication-Generation (SLURG) to address these limitations, exploring the feasibility of generating synthetic fallacious forum-style comments using large language models (LLMs), specifically DeepHermes-3-Mistral-24B. Our findings indicate that LLMs can replicate the syntactic patterns of real data} and that high-quality few-shot prompts enhance LLMs' ability to mimic the vocabulary diversity of online forums. 

---
# On Linear Representations and Pretraining Data Frequency in Language Models 

**Authors**: Jack Merullo, Noah A. Smith, Sarah Wiegreffe, Yanai Elazar  

**Link**: [PDF](https://arxiv.org/pdf/2504.12459)  

**Abstract**: Pretraining data has a direct impact on the behaviors and quality of language models (LMs), but we only understand the most basic principles of this relationship. While most work focuses on pretraining data's effect on downstream task behavior, we investigate its relationship to LM representations. Previous work has discovered that, in language models, some concepts are encoded `linearly' in the representations, but what factors cause these representations to form? We study the connection between pretraining data frequency and models' linear representations of factual relations. We find evidence that the formation of linear representations is strongly connected to pretraining term frequencies; specifically for subject-relation-object fact triplets, both subject-object co-occurrence frequency and in-context learning accuracy for the relation are highly correlated with linear representations. This is the case across all phases of pretraining. In OLMo-7B and GPT-J, we discover that a linear representation consistently (but not exclusively) forms when the subjects and objects within a relation co-occur at least 1k and 2k times, respectively, regardless of when these occurrences happen during pretraining. Finally, we train a regression model on measurements of linear representation quality in fully-trained LMs that can predict how often a term was seen in pretraining. Our model achieves low error even on inputs from a different model with a different pretraining dataset, providing a new method for estimating properties of the otherwise-unknown training data of closed-data models. We conclude that the strength of linear representations in LMs contains signal about the models' pretraining corpora that may provide new avenues for controlling and improving model behavior: particularly, manipulating the models' training data to meet specific frequency thresholds. 

---
# Position: The Most Expensive Part of an LLM should be its Training Data 

**Authors**: Nikhil Kandpal, Colin Raffel  

**Link**: [PDF](https://arxiv.org/pdf/2504.12427)  

**Abstract**: Training a state-of-the-art Large Language Model (LLM) is an increasingly expensive endeavor due to growing computational, hardware, energy, and engineering demands. Yet, an often-overlooked (and seldom paid) expense is the human labor behind these models' training data. Every LLM is built on an unfathomable amount of human effort: trillions of carefully written words sourced from books, academic papers, codebases, social media, and more. This position paper aims to assign a monetary value to this labor and argues that the most expensive part of producing an LLM should be the compensation provided to training data producers for their work. To support this position, we study 64 LLMs released between 2016 and 2024, estimating what it would cost to pay people to produce their training datasets from scratch. Even under highly conservative estimates of wage rates, the costs of these models' training datasets are 10-1000 times larger than the costs to train the models themselves, representing a significant financial liability for LLM providers. In the face of the massive gap between the value of training data and the lack of compensation for its creation, we highlight and discuss research directions that could enable fairer practices in the future. 

---
# A Method for Handling Negative Similarities in Explainable Graph Spectral Clustering of Text Documents -- Extended Version 

**Authors**: Mieczysław A. Kłopotek, Sławomir T. Wierzchoń, Bartłomiej Starosta, Dariusz Czerski, Piotr Borkowski  

**Link**: [PDF](https://arxiv.org/pdf/2504.12360)  

**Abstract**: This paper investigates the problem of Graph Spectral Clustering with negative similarities, resulting from document embeddings different from the traditional Term Vector Space (like doc2vec, GloVe, etc.). Solutions for combinatorial Laplacians and normalized Laplacians are discussed. An experimental investigation shows the advantages and disadvantages of 6 different solutions proposed in the literature and in this research. The research demonstrates that GloVe embeddings frequently cause failures of normalized Laplacian based GSC due to negative similarities. Furthermore, application of methods curing similarity negativity leads to accuracy improvement for both combinatorial and normalized Laplacian based GSC. It also leads to applicability for GloVe embeddings of explanation methods developed originally bythe authors for Term Vector Space embeddings. 

---
# Replicating ReLM Results: Validating Large Language Models with ReLM 

**Authors**: Reece Adamson, Erin Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.12357)  

**Abstract**: Validating Large Language Models with ReLM explores the application of formal languages to evaluate and control Large Language Models (LLMs) for memorization, bias, and zero-shot performance. Current approaches for evaluating these types behavior are often slow, imprecise, costly, or introduce biases of their own, but are necessary due to the importance of this behavior when productionizing LLMs. This project reproduces key results from the original ReLM paper and expounds on the approach and applications with an emphasis on the relevance to the field of systems for machine learning. 

---
# Leveraging Large Language Models for Multi-Class and Multi-Label Detection of Drug Use and Overdose Symptoms on Social Media 

**Authors**: Muhammad Ahmad, Muhammad Waqas, ldar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2504.12355)  

**Abstract**: Drug overdose remains a critical global health issue, often driven by misuse of opioids, painkillers, and psychiatric medications. Traditional research methods face limitations, whereas social media offers real-time insights into self-reported substance use and overdose symptoms. This study proposes an AI-driven NLP framework trained on annotated social media data to detect commonly used drugs and associated overdose symptoms. Using a hybrid annotation strategy with LLMs and human annotators, we applied traditional ML models, neural networks, and advanced transformer-based models. Our framework achieved 98% accuracy in multi-class and 97% in multi-label classification, outperforming baseline models by up to 8%. These findings highlight the potential of AI for supporting public health surveillance and personalized intervention strategies. 

---
# A Large-Language Model Framework for Relative Timeline Extraction from PubMed Case Reports 

**Authors**: Jing Wang, Jeremy C Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2504.12350)  

**Abstract**: Timing of clinical events is central to characterization of patient trajectories, enabling analyses such as process tracing, forecasting, and causal reasoning. However, structured electronic health records capture few data elements critical to these tasks, while clinical reports lack temporal localization of events in structured form. We present a system that transforms case reports into textual time series-structured pairs of textual events and timestamps. We contrast manual and large language model (LLM) annotations (n=320 and n=390 respectively) of ten randomly-sampled PubMed open-access (PMOA) case reports (N=152,974) and assess inter-LLM agreement (n=3,103; N=93). We find that the LLM models have moderate event recall(O1-preview: 0.80) but high temporal concordance among identified events (O1-preview: 0.95). By establishing the task, annotation, and assessment systems, and by demonstrating high concordance, this work may serve as a benchmark for leveraging the PMOA corpus for temporal analytics. 

---
# Mathematical Capabilities of Large Language Models in Finnish Matriculation Examination 

**Authors**: Mika Setälä, Pieta Sikström, Ville Heilala, Tommi Kärkkäinen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12347)  

**Abstract**: Large language models (LLMs) have shown increasing promise in educational settings, yet their mathematical reasoning has been considered evolving. This study evaluates the mathematical capabilities of various LLMs using the Finnish matriculation examination, a high-stakes digital test for upper secondary education. Initial tests yielded moderate performance corresponding to mid-range grades, but later evaluations demonstrated substantial improvements as the language models evolved. Remarkably, some models achieved near-perfect or perfect scores, matching top student performance and qualifying for university admission. Our findings highlight the rapid advances in the mathematical proficiency of LLMs and illustrate their potential to also support educational assessments at scale. 

---
# Reimagining Urban Science: Scaling Causal Inference with Large Language Models 

**Authors**: Yutong Xia, Ao Qu, Yunhan Zheng, Yihong Tang, Dingyi Zhuang, Yuxuan Liang, Cathy Wu, Roger Zimmermann, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12345)  

**Abstract**: Urban causal research is essential for understanding the complex dynamics of cities and informing evidence-based policies. However, it is challenged by the inefficiency and bias of hypothesis generation, barriers to multimodal data complexity, and the methodological fragility of causal experimentation. Recent advances in large language models (LLMs) present an opportunity to rethink how urban causal analysis is conducted. This Perspective examines current urban causal research by analyzing taxonomies that categorize research topics, data sources, and methodological approaches to identify structural gaps. We then introduce an LLM-driven conceptual framework, AutoUrbanCI, composed of four distinct modular agents responsible for hypothesis generation, data engineering, experiment design and execution, and results interpretation with policy recommendations. We propose evaluation criteria for rigor and transparency and reflect on implications for human-AI collaboration, equity, and accountability. We call for a new research agenda that embraces AI-augmented workflows not as replacements for human expertise but as tools to broaden participation, improve reproducibility, and unlock more inclusive forms of urban causal reasoning. 

---
# Propaganda via AI? A Study on Semantic Backdoors in Large Language Models 

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.12344)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at this https URL. 

---
# Benchmarking Biopharmaceuticals Retrieval-Augmented Generation Evaluation 

**Authors**: Hanmeng Zhong, Linqing Chen, Weilei Wang, Wentao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12342)  

**Abstract**: Recently, the application of the retrieval-augmented Large Language Models (LLMs) in specific domains has gained significant attention, especially in biopharmaceuticals. However, in this context, there is no benchmark specifically designed for biopharmaceuticals to evaluate LLMs. In this paper, we introduce the Biopharmaceuticals Retrieval-Augmented Generation Evaluation (BRAGE) , the first benchmark tailored for evaluating LLMs' Query and Reference Understanding Capability (QRUC) in the biopharmaceutical domain, available in English, French, German and Chinese. In addition, Traditional Question-Answering (QA) metrics like accuracy and exact match fall short in the open-ended retrieval-augmented QA scenarios. To address this, we propose a citation-based classification method to evaluate the QRUC of LLMs to understand the relationship between queries and references. We apply this method to evaluate the mainstream LLMs on BRAGE. Experimental results show that there is a significant gap in the biopharmaceutical QRUC of mainstream LLMs, and their QRUC needs to be improved. 

---
# Streamlining Biomedical Research with Specialized LLMs 

**Authors**: Linqing Chen, Weilei Wang, Yubin Xia, Wentao Wu, Peng Xu, Zilong Bai, Jie Fang, Chaobo Xu, Ran Hu, Licong Xu, Haoran Hua, Jing Sun, Hanmeng Zhong, Jin Liu, Tian Qiu, Haowen Liu, Meng Hu, Xiuwen Li, Fei Gao, Yong Gu, Tao Shi, Chaochao Wang, Jianping Lu, Cheng Sun, Yixin Wang, Shengjie Yang, Yuancheng Li, Lu Jin, Lisha Zhang, Fu Bian, Zhongkai Ye, Lidong Pei, Changyang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12341)  

**Abstract**: In this paper, we propose a novel system that integrates state-of-the-art, domain-specific large language models with advanced information retrieval techniques to deliver comprehensive and context-aware responses. Our approach facilitates seamless interaction among diverse components, enabling cross-validation of outputs to produce accurate, high-quality responses enriched with relevant data, images, tables, and other modalities. We demonstrate the system's capability to enhance response precision by leveraging a robust question-answering model, significantly improving the quality of dialogue generation. The system provides an accessible platform for real-time, high-fidelity interactions, allowing users to benefit from efficient human-computer interaction, precise retrieval, and simultaneous access to a wide range of literature and data. This dramatically improves the research efficiency of professionals in the biomedical and pharmaceutical domains and facilitates faster, more informed decision-making throughout the R\&D process. Furthermore, the system proposed in this paper is available at this https URL. 

---
# GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via A Dual-Branch Architecture 

**Authors**: Yaodong Song, Hongjie Chen, Jie Lian, Yuxin Zhang, Guangmin Xia, Zehan Li, Genliang Zhao, Jian Kang, Yongxiang Li, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12339)  

**Abstract**: While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-k layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data. 

---
# Paging Dr. GPT: Extracting Information from Clinical Notes to Enhance Patient Predictions 

**Authors**: David Anderson, Michaela Anderson, Margret Bjarnadottir, Stephen Mahar, Shriyan Reyya  

**Link**: [PDF](https://arxiv.org/pdf/2504.12338)  

**Abstract**: There is a long history of building predictive models in healthcare using tabular data from electronic medical records. However, these models fail to extract the information found in unstructured clinical notes, which document diagnosis, treatment, progress, medications, and care plans. In this study, we investigate how answers generated by GPT-4o-mini (ChatGPT) to simple clinical questions about patients, when given access to the patient's discharge summary, can support patient-level mortality prediction. Using data from 14,011 first-time admissions to the Coronary Care or Cardiovascular Intensive Care Units in the MIMIC-IV Note dataset, we implement a transparent framework that uses GPT responses as input features in logistic regression models. Our findings demonstrate that GPT-based models alone can outperform models trained on standard tabular data, and that combining both sources of information yields even greater predictive power, increasing AUC by an average of 5.1 percentage points and increasing positive predictive value by 29.9 percent for the highest-risk decile. These results highlight the value of integrating large language models (LLMs) into clinical prediction tasks and underscore the broader potential for using LLMs in any domain where unstructured text data remains an underutilized resource. 

---
# "It Listens Better Than My Therapist": Exploring Social Media Discourse on LLMs as Mental Health Tool 

**Authors**: Anna-Carolina Haensch  

**Link**: [PDF](https://arxiv.org/pdf/2504.12337)  

**Abstract**: The emergence of generative AI chatbots such as ChatGPT has prompted growing public and academic interest in their role as informal mental health support tools. While early rule-based systems have been around for several years, large language models (LLMs) offer new capabilities in conversational fluency, empathy simulation, and availability. This study explores how users engage with LLMs as mental health tools by analyzing over 10,000 TikTok comments from videos referencing LLMs as mental health tools. Using a self-developed tiered coding schema and supervised classification models, we identify user experiences, attitudes, and recurring themes. Results show that nearly 20% of comments reflect personal use, with these users expressing overwhelmingly positive attitudes. Commonly cited benefits include accessibility, emotional support, and perceived therapeutic value. However, concerns around privacy, generic responses, and the lack of professional oversight remain prominent. It is important to note that the user feedback does not indicate which therapeutic framework, if any, the LLM-generated output aligns with. While the findings underscore the growing relevance of AI in everyday practices, they also highlight the urgent need for clinical and ethical scrutiny in the use of AI for mental health support. 

---
# You've Changed: Detecting Modification of Black-Box Large Language Models 

**Authors**: Alden Dima, James Foulds, Shimei Pan, Philip Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2504.12335)  

**Abstract**: Large Language Models (LLMs) are often provided as a service via an API, making it challenging for developers to detect changes in their behavior. We present an approach to monitor LLMs for changes by comparing the distributions of linguistic and psycholinguistic features of generated text. Our method uses a statistical test to determine whether the distributions of features from two samples of text are equivalent, allowing developers to identify when an LLM has changed. We demonstrate the effectiveness of our approach using five OpenAI completion models and Meta's Llama 3 70B chat model. Our results show that simple text features coupled with a statistical test can distinguish between language models. We also explore the use of our approach to detect prompt injection attacks. Our work enables frequent LLM change monitoring and avoids computationally expensive benchmark evaluations. 

---
# QM-ToT: A Medical Tree of Thoughts Reasoning Framework for Quantized Model 

**Authors**: Zongxian Yang, Jiayu Qian, Zhi-An Huang, Kay Chen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12334)  

**Abstract**: Large language models (LLMs) face significant challenges in specialized biomedical tasks due to the inherent complexity of medical reasoning and the sensitive nature of clinical data. Existing LLMs often struggle with intricate medical terminology and the need for accurate clinical insights, leading to performance reduction when quantized for resource-constrained deployment. To address these issues, we propose Quantized Medical Tree of Thought (QM-ToT), a path-based reasoning framework. QM-ToT leverages a Tree of Thought (ToT) reasoning approach to decompose complex medical problems into manageable subtasks, coupled with evaluator assessment layers. This framework facilitates substantial performance improvements in INT4-quantized models on the challenging MedQAUSMLE dataset. Specifically, we demonstrate a remarkable accuracy increase from 34% to 50% for the LLaMA2-70b model and from 58.77% to 69.49% for LLaMA-3.1-8b. Besides, we also proposed an effect data distillation method based on ToT. Compared to the traditional distillation method, we achieved an improvement of 86. 27% while using only 3.9% of the this http URL work, for the first time, showcases the potential of ToT to significantly enhance performance on complex biomedical tasks, establishing a crucial foundation for future advances in deploying high-performing quantized LLM in resource-limited medical settings. 

---
# Meta-Evaluating Local LLMs: Rethinking Performance Metrics for Serious Games 

**Authors**: Andrés Isaza-Giraldo, Paulo Bala, Lucas Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2504.12333)  

**Abstract**: The evaluation of open-ended responses in serious games presents a unique challenge, as correctness is often subjective. Large Language Models (LLMs) are increasingly being explored as evaluators in such contexts, yet their accuracy and consistency remain uncertain, particularly for smaller models intended for local execution. This study investigates the reliability of five small-scale LLMs when assessing player responses in \textit{En-join}, a game that simulates decision-making within energy communities. By leveraging traditional binary classification metrics (including accuracy, true positive rate, and true negative rate), we systematically compare these models across different evaluation scenarios. Our results highlight the strengths and limitations of each model, revealing trade-offs between sensitivity, specificity, and overall performance. We demonstrate that while some models excel at identifying correct responses, others struggle with false positives or inconsistent evaluations. The findings highlight the need for context-aware evaluation frameworks and careful model selection when deploying LLMs as evaluators. This work contributes to the broader discourse on the trustworthiness of AI-driven assessment tools, offering insights into how different LLM architectures handle subjective evaluation tasks. 

---
# Can the capability of Large Language Models be described by human ability? A Meta Study 

**Authors**: Mingrui Zan, Yunquan Zhang, Boyang Zhang, Fangming Liu, Daning Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.12332)  

**Abstract**: Users of Large Language Models (LLMs) often perceive these models as intelligent entities with human-like capabilities. However, the extent to which LLMs' capabilities truly approximate human abilities remains a topic of debate. In this paper, to characterize the capabilities of LLMs in relation to human capabilities, we collected performance data from over 80 models across 37 evaluation benchmarks. The evaluation benchmarks are categorized into 6 primary abilities and 11 sub-abilities in human aspect. Then, we then clustered the performance rankings into several categories and compared these clustering results with classifications based on human ability aspects. Our findings lead to the following conclusions: 1. We have confirmed that certain capabilities of LLMs with fewer than 10 billion parameters can indeed be described using human ability metrics; 2. While some abilities are considered interrelated in humans, they appear nearly uncorrelated in LLMs; 3. The capabilities possessed by LLMs vary significantly with the parameter scale of the model. 

---
# Span-level Emotion-Cause-Category Triplet Extraction with Instruction Tuning LLMs and Data Augmentation 

**Authors**: Xiangju Li, Dong Yang, Xiaogang Zhu, Faliang Huang, Peng Zhang, Zhongying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12331)  

**Abstract**: Span-level emotion-cause-category triplet extraction represents a novel and complex challenge within emotion cause analysis. This task involves identifying emotion spans, cause spans, and their associated emotion categories within the text to form structured triplets. While prior research has predominantly concentrated on clause-level emotion-cause pair extraction and span-level emotion-cause detection, these methods often confront challenges originating from redundant information retrieval and difficulty in accurately determining emotion categories, particularly when emotions are expressed implicitly or ambiguously. To overcome these challenges, this study explores a fine-grained approach to span-level emotion-cause-category triplet extraction and introduces an innovative framework that leverages instruction tuning and data augmentation techniques based on large language models. The proposed method employs task-specific triplet extraction instructions and utilizes low-rank adaptation to fine-tune large language models, eliminating the necessity for intricate task-specific architectures. Furthermore, a prompt-based data augmentation strategy is developed to address data scarcity by guiding large language models in generating high-quality synthetic training data. Extensive experimental evaluations demonstrate that the proposed approach significantly outperforms existing baseline methods, achieving at least a 12.8% improvement in span-level emotion-cause-category triplet extraction metrics. The results demonstrate the method's effectiveness and robustness, offering a promising avenue for advancing research in emotion cause analysis. The source code is available at this https URL. 

---
# HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation 

**Authors**: Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12330)  

**Abstract**: While Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with external knowledge, conventional single-agent RAG remains fundamentally limited in resolving complex queries demanding coordinated reasoning across heterogeneous data ecosystems. We present HM-RAG, a novel Hierarchical Multi-agent Multimodal RAG framework that pioneers collaborative intelligence for dynamic knowledge synthesis across structured, unstructured, and graph-based data. The framework is composed of three-tiered architecture with specialized agents: a Decomposition Agent that dissects complex queries into contextually coherent sub-tasks via semantic-aware query rewriting and schema-guided context augmentation; Multi-source Retrieval Agents that carry out parallel, modality-specific retrieval using plug-and-play modules designed for vector, graph, and web-based databases; and a Decision Agent that uses consistency voting to integrate multi-source answers and resolve discrepancies in retrieval results through Expert Model Refinement. This architecture attains comprehensive query understanding by combining textual, graph-relational, and web-derived evidence, resulting in a remarkable 12.95% improvement in answer accuracy and a 3.56% boost in question classification accuracy over baseline RAG systems on the ScienceQA and CrisisMMD benchmarks. Notably, HM-RAG establishes state-of-the-art results in zero-shot settings on both datasets. Its modular architecture ensures seamless integration of new data modalities while maintaining strict data governance, marking a significant advancement in addressing the critical challenges of multimodal reasoning and knowledge synthesis in RAG systems. Code is available at this https URL. 

---
# Speculative Thinking: Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time 

**Authors**: Wang Yang, Xiang Yue, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.12329)  

**Abstract**: Recent advances leverage post-training to enhance model reasoning performance, which typically requires costly training pipelines and still suffers from inefficient, overly lengthy outputs. We introduce Speculative Thinking, a training-free framework that enables large reasoning models to guide smaller ones during inference at the reasoning level, distinct from speculative decoding, which operates at the token level. Our approach is based on two observations: (1) reasoning-supportive tokens such as "wait" frequently appear after structural delimiters like "\n\n", serving as signals for reflection or continuation; and (2) larger models exhibit stronger control over reflective behavior, reducing unnecessary backtracking while improving reasoning quality. By strategically delegating reflective steps to a more capable model, our method significantly boosts the reasoning accuracy of reasoning models while shortening their output. With the assistance of the 32B reasoning model, the 1.5B model's accuracy on MATH500 increases from 83.2% to 89.4%, marking a substantial improvement of 6.2%. Simultaneously, the average output length is reduced from 5439 tokens to 4583 tokens, representing a 15.7% decrease. Moreover, when applied to a non-reasoning model (Qwen-2.5-7B-Instruct), our framework boosts its accuracy from 74.0% to 81.8% on the same benchmark, achieving a relative improvement of 7.8%. 

---
# A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future 

**Authors**: Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jinjie Gu, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.12328)  

**Abstract**: Reward Model (RM) has demonstrated impressive potential for enhancing Large Language Models (LLM), as RM can serve as a proxy for human preferences, providing signals to guide LLMs' behavior in various tasks. In this paper, we provide a comprehensive overview of relevant research, exploring RMs from the perspectives of preference collection, reward modeling, and usage. Next, we introduce the applications of RMs and discuss the benchmarks for evaluation. Furthermore, we conduct an in-depth analysis of the challenges existing in the field and dive into the potential research directions. This paper is dedicated to providing beginners with a comprehensive introduction to RMs and facilitating future studies. The resources are publicly available at github\footnote{this https URL}. 

---
# Word Embeddings Track Social Group Changes Across 70 Years in China 

**Authors**: Yuxi Ma, Yongqian Peng, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12327)  

**Abstract**: Language encodes societal beliefs about social groups through word patterns. While computational methods like word embeddings enable quantitative analysis of these patterns, studies have primarily examined gradual shifts in Western contexts. We present the first large-scale computational analysis of Chinese state-controlled media (1950-2019) to examine how revolutionary social transformations are reflected in official linguistic representations of social groups. Using diachronic word embeddings at multiple temporal resolutions, we find that Chinese representations differ significantly from Western counterparts, particularly regarding economic status, ethnicity, and gender. These representations show distinct evolutionary dynamics: while stereotypes of ethnicity, age, and body type remain remarkably stable across political upheavals, representations of gender and economic classes undergo dramatic shifts tracking historical transformations. This work advances our understanding of how officially sanctioned discourse encodes social structure through language while highlighting the importance of non-Western perspectives in computational social science. 

---
# Reconstructing Sepsis Trajectories from Clinical Case Reports using LLMs: the Textual Time Series Corpus for Sepsis 

**Authors**: Shahriar Noroozizadeh, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2504.12326)  

**Abstract**: Clinical case reports and discharge summaries may be the most complete and accurate summarization of patient encounters, yet they are finalized, i.e., timestamped after the encounter. Complementary data structured streams become available sooner but suffer from incompleteness. To train models and algorithms on more complete and temporally fine-grained data, we construct a pipeline to phenotype, extract, and annotate time-localized findings within case reports using large language models. We apply our pipeline to generate an open-access textual time series corpus for Sepsis-3 comprising 2,139 case reports from the Pubmed-Open Access (PMOA) Subset. To validate our system, we apply it on PMOA and timeline annotations from I2B2/MIMIC-IV and compare the results to physician-expert annotations. We show high recovery rates of clinical findings (event match rates: O1-preview--0.755, Llama 3.3 70B Instruct--0.753) and strong temporal ordering (concordance: O1-preview--0.932, Llama 3.3 70B Instruct--0.932). Our work characterizes the ability of LLMs to time-localize clinical findings in text, illustrating the limitations of LLM use for temporal reconstruction and providing several potential avenues of improvement via multimodal integration. 

---
# LLMTaxo: Leveraging Large Language Models for Constructing Taxonomy of Factual Claims from Social Media 

**Authors**: Haiqi Zhang, Zhengyuan Zhu, Zeyu Zhang, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12325)  

**Abstract**: With the vast expansion of content on social media platforms, analyzing and comprehending online discourse has become increasingly complex. This paper introduces LLMTaxo, a novel framework leveraging large language models for the automated construction of taxonomy of factual claims from social media by generating topics from multi-level granularities. This approach aids stakeholders in more effectively navigating the social media landscapes. We implement this framework with different models across three distinct datasets and introduce specially designed taxonomy evaluation metrics for a comprehensive assessment. With the evaluations from both human evaluators and GPT-4, the results indicate that LLMTaxo effectively categorizes factual claims from social media, and reveals that certain models perform better on specific datasets. 

---
# Cross-Document Cross-Lingual Natural Language Inference via RST-enhanced Graph Fusion and Interpretability Prediction 

**Authors**: Mengying Yuan, Wangzi Xuan, Fei Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12324)  

**Abstract**: Natural Language Inference (NLI) is a fundamental task in both natural language processing and information retrieval. While NLI has developed many sub-directions such as sentence-level NLI, document-level NLI and cross-lingual NLI, Cross-Document Cross-Lingual NLI (CDCL-NLI) remains largely unexplored. In this paper, we propose a novel paradigm for CDCL-NLI that extends traditional NLI capabilities to multi-document, multilingual scenarios. To support this task, we construct a high-quality CDCL-NLI dataset including 1,110 instances and spanning 26 languages. To build a baseline for this task, we also propose an innovative method that integrates RST-enhanced graph fusion and interpretability prediction. Our method employs RST (Rhetorical Structure Theory) on RGAT (Relation-aware Graph Attention Network) for cross-document context modeling, coupled with a structure-aware semantic alignment mechanism based on lexical chains for cross-lingual understanding. For NLI interpretability, we develop an EDU-level attribution framework that generates extractive explanations. Extensive experiments demonstrate our approach's superior performance, achieving significant improvements over both traditional NLI models such as DocNLI and R2F, as well as LLMs like Llama3 and GPT-4o. Our work sheds light on the study of NLI and will bring research interest on cross-document cross-lingual context understanding, semantic retrieval and interpretability inference. Our dataset and code are available at \href{this https URL}{CDCL-NLI-Link for peer review}. 

---
# The Other Side of the Coin: Exploring Fairness in Retrieval-Augmented Generation 

**Authors**: Zheng Zhang, Ning Li, Qi Liu, Rui Li, Weibo Gao, Qingyang Mao, Zhenya Huang, Baosheng Yu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12323)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant document from external knowledge sources. By referencing this external knowledge, RAG effectively reduces the generation of factually incorrect content and addresses hallucination issues within LLMs. Recently, there has been growing attention to improving the performance and efficiency of RAG systems from various perspectives. While these advancements have yielded significant results, the application of RAG in domains with considerable societal implications raises a critical question about fairness: What impact does the introduction of the RAG paradigm have on the fairness of LLMs? To address this question, we conduct extensive experiments by varying the LLMs, retrievers, and retrieval sources. Our experimental analysis reveals that the scale of the LLMs plays a significant role in influencing fairness outcomes within the RAG framework. When the model scale is smaller than 8B, the integration of retrieval mechanisms often exacerbates unfairness in small-scale LLMs (e.g., LLaMA3.2-1B, Mistral-7B, and LLaMA3-8B). To mitigate the fairness issues introduced by RAG for small-scale LLMs, we propose two approaches, FairFT and FairFilter. Specifically, in FairFT, we align the retriever with the LLM in terms of fairness, enabling it to retrieve documents that facilitate fairer model outputs. In FairFilter, we propose a fairness filtering mechanism to filter out biased content after retrieval. Finally, we validate our proposed approaches on real-world datasets, demonstrating their effectiveness in improving fairness while maintaining performance. 

---
# A Strategic Coordination Framework of Small LLMs Matches Large LLMs in Data Synthesis 

**Authors**: Xin Gao, Qizhi Pei, Zinan Tang, Yu Li, Honglin Lin, Jiang Wu, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12322)  

**Abstract**: While data synthesis and distillation are promising strategies to enhance small language models, current approaches heavily rely on Large Language Models (LLMs), which suffer from high computational costs, environmental inefficiency, and potential biases inherited from monolithic architectures. In contrast, smaller LLMs are more accessible and sustainable, but their individual capabilities often fall short in generating high-quality, diverse, and reliable data. Inspired by collaborative human processes (e.g., peer review), we propose a multiple small LLMs involved framework, GRA, that aggregates specialized roles across small LLMs to iterative refinement and quality control typically achieved by a single large LLM. In this collaborative framework, multiple small LLMs assume distinct roles-Generator, Reviewer, and Adjudicator-to simulate a peer-review-inspired data synthesis pipeline. The Generator proposes initial data samples, the Reviewer critiques their quality and diversity, and the Adjudicator resolves conflicts to finalize the output. By decomposing the synthesis process into specialized sub-tasks, collaborative small LLMs can achieve data-level parity with large LLM-based distillation. Through experiments across multiple benchmarks, we demonstrate that GRA-produced data matches or exceeds the quality of single large LLM outputs, e.g., Qwen-2.5-72B-Instruct. Our results challenge the necessity of monolithic large models for high-quality data synthesis, advocating instead for strategic coordination of smaller agents. Our datasets, models, and code are publicly available at this https URL. 

---
# AttentionDefense: Leveraging System Prompt Attention for Explainable Defense Against Novel Jailbreaks 

**Authors**: Charlotte Siska, Anush Sankaran  

**Link**: [PDF](https://arxiv.org/pdf/2504.12321)  

**Abstract**: In the past few years, Language Models (LMs) have shown par-human capabilities in several domains. Despite their practical applications and exceeding user consumption, they are susceptible to jailbreaks when malicious input exploits the LM's weaknesses, causing it to deviate from its intended behavior. Current defensive strategies either classify the input prompt as adversarial or prevent LMs from generating harmful outputs. However, it is challenging to explain the reason behind the malicious nature of the jailbreak, which results in a wide variety of closed-box approaches. In this research, we propose and demonstrate that system-prompt attention from Small Language Models (SLMs) can be used to characterize adversarial prompts, providing a novel, explainable, and cheaper defense approach called AttentionDefense. Our research suggests that the attention mechanism is an integral component in understanding and explaining how LMs respond to malicious input that is not captured in the semantic meaning of text embeddings. The proposed AttentionDefense is evaluated against existing jailbreak benchmark datasets. Ablation studies show that SLM-based AttentionDefense has equivalent or better jailbreak detection performance compared to text embedding-based classifiers and GPT-4 zero-shot this http URL further validate the efficacy of the proposed approach, we generate a dataset of novel jailbreak variants of the existing benchmark dataset using a closed-loop LLM-based multi-agent system. We demonstrate that the proposed AttentionDefense approach performs robustly on this novel jailbreak dataset while existing approaches suffer in performance. Additionally, for practical purposes AttentionDefense is an ideal solution as it has the computation requirements of a small LM but the performance of a LLM detector. 

---
# Has the Creativity of Large-Language Models peaked? An analysis of inter- and intra-LLM variability 

**Authors**: Jennifer Haase, Paul H. P. Hanel, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2504.12320)  

**Abstract**: Following the widespread adoption of ChatGPT in early 2023, numerous studies reported that large language models (LLMs) can match or even surpass human performance in creative tasks. However, it remains unclear whether LLMs have become more creative over time, and how consistent their creative output is. In this study, we evaluated 14 widely used LLMs -- including GPT-4, Claude, Llama, Grok, Mistral, and DeepSeek -- across two validated creativity assessments: the Divergent Association Task (DAT) and the Alternative Uses Task (AUT). Contrary to expectations, we found no evidence of increased creative performance over the past 18-24 months, with GPT-4 performing worse than in previous studies. For the more widely used AUT, all models performed on average better than the average human, with GPT-4o and o3-mini performing best. However, only 0.28% of LLM-generated responses reached the top 10% of human creativity benchmarks. Beyond inter-model differences, we document substantial intra-model variability: the same LLM, given the same prompt, can produce outputs ranging from below-average to original. This variability has important implications for both creativity research and practical applications. Ignoring such variability risks misjudging the creative potential of LLMs, either inflating or underestimating their capabilities. The choice of prompts affected LLMs differently. Our findings underscore the need for more nuanced evaluation frameworks and highlight the importance of model selection, prompt design, and repeated assessment when using Generative AI (GenAI) tools in creative contexts. 

---
# ChatGPT as Linguistic Equalizer? Quantifying LLM-Driven Lexical Shifts in Academic Writing 

**Authors**: Dingkang Lin, Naixuan Zhao, Dan Tian, Jiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12317)  

**Abstract**: The advent of ChatGPT has profoundly reshaped scientific research practices, particularly in academic writing, where non-native English-speakers (NNES) historically face linguistic barriers. This study investigates whether ChatGPT mitigates these barriers and fosters equity by analyzing lexical complexity shifts across 2.8 million articles from OpenAlex (2020-2024). Using the Measure of Textual Lexical Diversity (MTLD) to quantify vocabulary sophistication and a difference-in-differences (DID) design to identify causal effects, we demonstrate that ChatGPT significantly enhances lexical complexity in NNES-authored abstracts, even after controlling for article-level controls, authorship patterns, and venue norms. Notably, the impact is most pronounced in preprint papers, technology- and biology-related fields and lower-tier journals. These findings provide causal evidence that ChatGPT reduces linguistic disparities and promotes equity in global academia. 

---
# Data Metabolism: An Efficient Data Design Schema For Vision Language Model 

**Authors**: Jingyuan Zhang, Hongzhi Zhang, Zhou Haonan, Chenxi Sun, Xingguang ji, Jiakang Wang, Fanheng Kong, Yahui Liu, Qi Wang, Fuzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12316)  

**Abstract**: Data curation plays a crucial role in training powerful Visual Language Models (VLMs). In this work, we introduce the concept of Data Metabolism and present our data-centric framework to build VLMs throughout the development lifecycle. Starting from a standard model architecture, we discuss and provide insights into two crucial development steps: data curation and iteration, forming a closed-loop system that continuously improves model performance. We show a detailed codebook on how to process existing massive datasets and build user-specific data flywheel. As a demonstration, we release a VLM, named Capybara-VL, which excels in typical multimodal tasks (e.g. , visual question answering, scientific reasoning, and text-rich tasks). Despite its relatively compact size, Capybara-VL surpasses several open-source models that are up to 10 times larger in size. Moreover, it achieves results that are on par with those of several leading proprietary models, demonstrating its remarkable competitiveness. These results highlight the power of our data-centric framework and the potential of training smaller and more efficient VLMs. 

---
# Capybara-OMNI: An Efficient Paradigm for Building Omni-Modal Language Models 

**Authors**: Xingguang Ji, Jiakang Wang, Hongzhi Zhang, Jingyuan Zhang, Haonan Zhou, Chenxi Sun, Yahui Liu, Qi Wang, Fuzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12315)  

**Abstract**: With the development of Multimodal Large Language Models (MLLMs), numerous outstanding accomplishments have emerged within the open-source community. Due to the complexity of creating and training multimodal data pairs, it is still a computational and time-consuming process to build powerful MLLMs. In this work, we introduce Capybara-OMNI, an MLLM that trains in a lightweight and efficient manner and supports understanding text, image, video, and audio modalities. We present in detail the framework design, the data construction, and the training recipe, to develop an MLLM step-by-step to obtain competitive performance. We also provide exclusive benchmarks utilized in our experiments to show how to properly verify understanding capabilities across different modalities. Results show that by following our guidance, we can efficiently build an MLLM that achieves competitive performance among models of the same scale on various multimodal benchmarks. Additionally, to enhance the multimodal instruction following and conversational capabilities of the model, we further discuss how to train the chat version upon an MLLM understanding model, which is more in line with user habits for tasks like real-time interaction with humans. We publicly disclose the Capybara-OMNI model, along with its chat-based version. The disclosure includes both the model weights, a portion of the training data, and the inference codes, which are made available on GitHub. 

---
# How to Detect and Defeat Molecular Mirage: A Metric-Driven Benchmark for Hallucination in LLM-based Molecular Comprehension 

**Authors**: Hao Li, Liuzhenghao Lv, He Cao, Zijing Liu, Zhiyuan Yan, Yu Wang, Yonghong Tian, Yu Li, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12314)  

**Abstract**: Large language models are increasingly used in scientific domains, especially for molecular understanding and analysis. However, existing models are affected by hallucination issues, resulting in errors in drug design and utilization. In this paper, we first analyze the sources of hallucination in LLMs for molecular comprehension tasks, specifically the knowledge shortcut phenomenon observed in the PubChem dataset. To evaluate hallucination in molecular comprehension tasks with computational efficiency, we introduce \textbf{Mol-Hallu}, a novel free-form evaluation metric that quantifies the degree of hallucination based on the scientific entailment relationship between generated text and actual molecular properties. Utilizing the Mol-Hallu metric, we reassess and analyze the extent of hallucination in various LLMs performing molecular comprehension tasks. Furthermore, the Hallucination Reduction Post-processing stage~(HRPP) is proposed to alleviate molecular hallucinations, Experiments show the effectiveness of HRPP on decoder-only and encoder-decoder molecular LLMs. Our findings provide critical insights into mitigating hallucination and improving the reliability of LLMs in scientific applications. 

---
# Exploring the Impact of Personality Traits on Conversational Recommender Systems: A Simulation with Large Language Models 

**Authors**: Xiaoyan Zhao, Yang Deng, Wenjie Wang, Hongzhan lin, Hong Cheng, Rui Zhang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.12313)  

**Abstract**: Conversational Recommender Systems (CRSs) engage users in multi-turn interactions to deliver personalized recommendations. The emergence of large language models (LLMs) further enhances these systems by enabling more natural and dynamic user interactions. However, a key challenge remains in understanding how personality traits shape conversational recommendation outcomes. Psychological evidence highlights the influence of personality traits on user interaction behaviors. To address this, we introduce an LLM-based personality-aware user simulation for CRSs (PerCRS). The user agent induces customizable personality traits and preferences, while the system agent possesses the persuasion capability to simulate realistic interaction in CRSs. We incorporate multi-aspect evaluation to ensure robustness and conduct extensive analysis from both user and system perspectives. Experimental results demonstrate that state-of-the-art LLMs can effectively generate diverse user responses aligned with specified personality traits, thereby prompting CRSs to dynamically adjust their recommendation strategies. Our experimental analysis offers empirical insights into the impact of personality traits on the outcomes of conversational recommender systems. 

---
# Socrates or Smartypants: Testing Logic Reasoning Capabilities of Large Language Models with Logic Programming-based Test Oracles 

**Authors**: Zihao Xu, Junchen Ding, Yiling Lou, Kun Zhang, Dong Gong, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12312)  

**Abstract**: Large Language Models (LLMs) have achieved significant progress in language understanding and reasoning. Evaluating and analyzing their logical reasoning abilities has therefore become essential. However, existing datasets and benchmarks are often limited to overly simplistic, unnatural, or contextually constrained examples. In response to the growing demand, we introduce SmartyPat-Bench, a challenging, naturally expressed, and systematically labeled benchmark derived from real-world high-quality Reddit posts containing subtle logical fallacies. Unlike existing datasets and benchmarks, it provides more detailed annotations of logical fallacies and features more diverse data. To further scale up the study and address the limitations of manual data collection and labeling - such as fallacy-type imbalance and labor-intensive annotation - we introduce SmartyPat, an automated framework powered by logic programming-based oracles. SmartyPat utilizes Prolog rules to systematically generate logically fallacious statements, which are then refined into fluent natural-language sentences by LLMs, ensuring precise fallacy representation. Extensive evaluation demonstrates that SmartyPat produces fallacies comparable in subtlety and quality to human-generated content and significantly outperforms baseline methods. Finally, experiments reveal nuanced insights into LLM capabilities, highlighting that while excessive reasoning steps hinder fallacy detection accuracy, structured reasoning enhances fallacy categorization performance. 

---
# Learning Optimal Prompt Ensemble for Multi-source Visual Prompt Transfer 

**Authors**: Enming Zhang, Liwen Cao, Yanru Wu, Zijie Zhao, Guan Wang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12311)  

**Abstract**: Prompt tuning has emerged as a lightweight adaptation strategy for adapting foundation models to downstream tasks, particularly in resource-constrained systems. As pre-trained prompts have become valuable intellectual assets, combining multiple source prompts offers a promising approach to enhance generalization to new tasks by leveraging complementary knowledge from diverse sources. However, naive aggregation of these prompts often leads to representation collapse due to mutual interference, undermining their collective potential. To address these challenges, we propose HGPrompt, an adaptive framework for multi-source prompt transfer that learns optimal ensemble weights by jointly optimizing dual objectives: transferability and stability. Specifically, we first introduce an information-theoretic metric to evaluate the transferability of prompt-induced features on the target task, capturing the intrinsic alignment between the feature representations. Additionally, we propose a novel Gradient Alignment Regularization to mitigate gradient conflicts among prompts, enabling stable and coherent knowledge transfer from multiple sources while suppressing interference. Extensive experiments on the large-scale VTAB benchmark demonstrate that HGPrompt achieves state-of-the-art performance, validating its effectiveness in multi-source prompt transfer. 

---
# Unmasking the Reality of PII Masking Models: Performance Gaps and the Call for Accountability 

**Authors**: Devansh Singh, Sundaraparipurnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12308)  

**Abstract**: Privacy Masking is a critical concept under data privacy involving anonymization and de-anonymization of personally identifiable information (PII). Privacy masking techniques rely on Named Entity Recognition (NER) approaches under NLP support in identifying and classifying named entities in each text. NER approaches, however, have several limitations including (a) content sensitivity including ambiguous, polysemic, context dependent or domain specific content, (b) phrasing variabilities including nicknames and alias, informal expressions, alternative representations, emerging expressions, evolving naming conventions and (c) formats or syntax variations, typos, misspellings. However, there are a couple of PII datasets that have been widely used by researchers and the open-source community to train models on PII detection or masking. These datasets have been used to train models including Piiranha and Starpii, which have been downloaded over 300k and 580k times on HuggingFace. We examine the quality of the PII masking by these models given the limitations of the datasets and of the NER approaches. We curate a dataset of 17K unique, semi-synthetic sentences containing 16 types of PII by compiling information from across multiple jurisdictions including India, U.K and U.S. We generate sentences (using language models) containing these PII at five different NER detection feature dimensions - (1) Basic Entity Recognition, (2) Contextual Entity Disambiguation, (3) NER in Noisy & Real-World Data, (4) Evolving & Novel Entities Detection and (5) Cross-Lingual or multi-lingual NER) and 1 in adversarial context. We present the results and exhibit the privacy exposure caused by such model use (considering the extent of lifetime downloads of these models). We conclude by highlighting the gaps in measuring performance of the models and the need for contextual disclosure in model cards for such models. 

---
# SemCORE: A Semantic-Enhanced Generative Cross-Modal Retrieval Framework with MLLMs 

**Authors**: Haoxuan Li, Yi Bin, Yunshan Ma, Guoqing Wang, Yang Yang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.13172)  

**Abstract**: Cross-modal retrieval (CMR) is a fundamental task in multimedia research, focused on retrieving semantically relevant targets across different modalities. While traditional CMR methods match text and image via embedding-based similarity calculations, recent advancements in pre-trained generative models have established generative retrieval as a promising alternative. This paradigm assigns each target a unique identifier and leverages a generative model to directly predict identifiers corresponding to input queries without explicit indexing. Despite its great potential, current generative CMR approaches still face semantic information insufficiency in both identifier construction and generation processes. To address these limitations, we propose a novel unified Semantic-enhanced generative Cross-mOdal REtrieval framework (SemCORE), designed to unleash the semantic understanding capabilities in generative cross-modal retrieval task. Specifically, we first construct a Structured natural language IDentifier (SID) that effectively aligns target identifiers with generative models optimized for natural language comprehension and generation. Furthermore, we introduce a Generative Semantic Verification (GSV) strategy enabling fine-grained target discrimination. Additionally, to the best of our knowledge, SemCORE is the first framework to simultaneously consider both text-to-image and image-to-text retrieval tasks within generative cross-modal retrieval. Extensive experiments demonstrate that our framework outperforms state-of-the-art generative cross-modal retrieval methods. Notably, SemCORE achieves substantial improvements across benchmark datasets, with an average increase of 8.65 points in Recall@1 for text-to-image retrieval. 

---
# Sleep-time Compute: Beyond Inference Scaling at Test-time 

**Authors**: Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2504.13171)  

**Abstract**: Scaling test-time compute has emerged as a key ingredient for enabling large language models (LLMs) to solve difficult problems, but comes with high latency and inference cost. We introduce sleep-time compute, which allows models to "think" offline about contexts before queries are presented: by anticipating what queries users might ask and pre-computing useful quantities, we can significantly reduce the compute requirements at test-time. To demonstrate the efficacy of our method, we create modified versions of two reasoning tasks - Stateful GSM-Symbolic and Stateful AIME. We find that sleep-time compute can reduce the amount of test-time compute needed to achieve the same accuracy by ~ 5x on Stateful GSM-Symbolic and Stateful AIME and that by scaling sleep-time compute we can further increase accuracy by up to 13% on Stateful GSM-Symbolic and 18% on Stateful AIME. Furthermore, we introduce Multi-Query GSM-Symbolic, which extends GSM-Symbolic by including multiple related queries per context. By amortizing sleep-time compute across related queries about the same context using Multi-Query GSM-Symbolic, we can decrease the average cost per query by 2.5x. We then conduct additional analysis to understand when sleep-time compute is most effective, finding the predictability of the user query to be well correlated with the efficacy of sleep-time compute. Finally, we conduct a case-study of applying sleep-time compute to a realistic agentic SWE task. 

---
# MIB: A Mechanistic Interpretability Benchmark 

**Authors**: Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, Dana Arad, Iván Arcuschin, Adam Belfki, Yik Siu Chan, Jaden Fiotto-Kaufman, Tal Haklay, Michael Hanna, Jing Huang, Rohan Gupta, Yaniv Nikankin, Hadas Orgad, Nikhil Prakash, Anja Reusch, Aruna Sankaranarayanan, Shun Shao, Alessandro Stolfo, Martin Tutek, Amir Zur, David Bau, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2504.13151)  

**Abstract**: How can we know whether new mechanistic interpretability methods achieve real improvements? In pursuit of meaningful and lasting evaluation standards, we propose MIB, a benchmark with two tracks spanning four tasks and five models. MIB favors methods that precisely and concisely recover relevant causal pathways or specific causal variables in neural language models. The circuit localization track compares methods that locate the model components - and connections between them - most important for performing a task (e.g., attribution patching or information flow routes). The causal variable localization track compares methods that featurize a hidden vector, e.g., sparse autoencoders (SAEs) or distributed alignment search (DAS), and locate model features for a causal variable relevant to the task. Using MIB, we find that attribution and mask optimization methods perform best on circuit localization. For causal variable localization, we find that the supervised DAS method performs best, while SAE features are not better than neurons, i.e., standard dimensions of hidden vectors. These findings illustrate that MIB enables meaningful comparisons of methods, and increases our confidence that there has been real progress in the field. 

---
# Antidistillation Sampling 

**Authors**: Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2504.13146)  

**Abstract**: Frontier models that generate extended reasoning traces inadvertently produce rich token sequences that can facilitate model distillation. Recognizing this vulnerability, model owners may seek sampling strategies that limit the effectiveness of distillation without compromising model performance. \emph{Antidistillation sampling} provides exactly this capability. By strategically modifying a model's next-token probability distribution, antidistillation sampling poisons reasoning traces, rendering them significantly less effective for distillation while preserving the model's practical utility. For further details, see this https URL. 

---
# FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents 

**Authors**: Nandan Thakur, Jimmy Lin, Sam Havens, Michael Carbin, Omar Khattab, Andrew Drozdov  

**Link**: [PDF](https://arxiv.org/pdf/2504.13128)  

**Abstract**: We introduce FreshStack, a reusable framework for automatically building information retrieval (IR) evaluation benchmarks from community-asked questions and answers. FreshStack conducts the following steps: (1) automatic corpus collection from code and technical documentation, (2) nugget generation from community-asked questions and answers, and (3) nugget-level support, retrieving documents using a fusion of retrieval techniques and hybrid architectures. We use FreshStack to build five datasets on fast-growing, recent, and niche topics to ensure the tasks are sufficiently challenging. On FreshStack, existing retrieval models, when applied out-of-the-box, significantly underperform oracle approaches on all five topics, denoting plenty of headroom to improve IR quality. In addition, we identify cases where rerankers do not clearly improve first-stage retrieval accuracy (two out of five topics). We hope that FreshStack will facilitate future work toward constructing realistic, scalable, and uncontaminated IR and RAG evaluation benchmarks. FreshStack datasets are available at: this https URL. 

---
# Probing and Inducing Combinational Creativity in Vision-Language Models 

**Authors**: Yongqian Peng, Yuxi Ma, Mengmeng Wang, Yuxuan Wang, Yizhou Wang, Chi Zhang, Yixin Zhu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.13120)  

**Abstract**: The ability to combine existing concepts into novel ideas stands as a fundamental hallmark of human intelligence. Recent advances in Vision-Language Models (VLMs) like GPT-4V and DALLE-3 have sparked debate about whether their outputs reflect combinational creativity--defined by M. A. Boden (1998) as synthesizing novel ideas through combining existing concepts--or sophisticated pattern matching of training data. Drawing inspiration from cognitive science, we investigate the combinational creativity of VLMs from the lens of concept blending. We propose the Identification-Explanation-Implication (IEI) framework, which decomposes creative processes into three levels: identifying input spaces, extracting shared attributes, and deriving novel semantic implications. To validate this framework, we curate CreativeMashup, a high-quality dataset of 666 artist-generated visual mashups annotated according to the IEI framework. Through extensive experiments, we demonstrate that in comprehension tasks, best VLMs have surpassed average human performance while falling short of expert-level understanding; in generation tasks, incorporating our IEI framework into the generation pipeline significantly enhances the creative quality of VLMs outputs. Our findings establish both a theoretical foundation for evaluating artificial creativity and practical guidelines for improving creative generation in VLMs. 

---
# Tackling Social Bias against the Poor: A Dataset and Taxonomy on Aporophobia 

**Authors**: Georgina Curto, Svetlana Kiritchenko, Muhammad Hammad Fahim Siddiqui, Isar Nejadgholi, Kathleen C. Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2504.13085)  

**Abstract**: Eradicating poverty is the first goal in the United Nations Sustainable Development Goals. However, aporophobia -- the societal bias against people living in poverty -- constitutes a major obstacle to designing, approving and implementing poverty-mitigation policies. This work presents an initial step towards operationalizing the concept of aporophobia to identify and track harmful beliefs and discriminative actions against poor people on social media. In close collaboration with non-profits and governmental organizations, we conduct data collection and exploration. Then we manually annotate a corpus of English tweets from five world regions for the presence of (1) direct expressions of aporophobia, and (2) statements referring to or criticizing aporophobic views or actions of others, to comprehensively characterize the social media discourse related to bias and discrimination against the poor. Based on the annotated data, we devise a taxonomy of categories of aporophobic attitudes and actions expressed through speech on social media. Finally, we train several classifiers and identify the main challenges for automatic detection of aporophobia in social networks. This work paves the way towards identifying, tracking, and mitigating aporophobic views on social media at scale. 

---
# RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins 

**Authors**: Yao Mu, Tianxing Chen, Zanxin Chen, Shijia Peng, Zhiqian Lan, Zeyu Gao, Zhixuan Liang, Qiaojun Yu, Yude Zou, Mingkun Xu, Lunkai Lin, Zhiqiang Xie, Mingyu Ding, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.13059)  

**Abstract**: In the rapidly advancing field of robotics, dual-arm coordination and complex object manipulation are essential capabilities for developing advanced autonomous systems. However, the scarcity of diverse, high-quality demonstration data and real-world-aligned evaluation benchmarks severely limits such development. To address this, we introduce RoboTwin, a generative digital twin framework that uses 3D generative foundation models and large language models to produce diverse expert datasets and provide a real-world-aligned evaluation platform for dual-arm robotic tasks. Specifically, RoboTwin creates varied digital twins of objects from single 2D images, generating realistic and interactive scenarios. It also introduces a spatial relation-aware code generation framework that combines object annotations with large language models to break down tasks, determine spatial constraints, and generate precise robotic movement code. Our framework offers a comprehensive benchmark with both simulated and real-world data, enabling standardized evaluation and better alignment between simulated training and real-world performance. We validated our approach using the open-source COBOT Magic Robot platform. Policies pre-trained on RoboTwin-generated data and fine-tuned with limited real-world samples demonstrate significant potential for enhancing dual-arm robotic manipulation systems by improving success rates by over 70% for single-arm tasks and over 40% for dual-arm tasks compared to models trained solely on real-world data. 

---
# How Large Language Models Are Changing MOOC Essay Answers: A Comparison of Pre- and Post-LLM Responses 

**Authors**: Leo Leppänen, Lili Aunimo, Arto Hellas, Jukka K. Nurminen, Linda Mannila  

**Link**: [PDF](https://arxiv.org/pdf/2504.13038)  

**Abstract**: The release of ChatGPT in late 2022 caused a flurry of activity and concern in the academic and educational communities. Some see the tool's ability to generate human-like text that passes at least cursory inspections for factual accuracy ``often enough'' a golden age of information retrieval and computer-assisted learning. Some, on the other hand, worry the tool may lead to unprecedented levels of academic dishonesty and cheating. In this work, we quantify some of the effects of the emergence of Large Language Models (LLMs) on online education by analyzing a multi-year dataset of student essay responses from a free university-level MOOC on AI ethics. Our dataset includes essays submitted both before and after ChatGPT's release. We find that the launch of ChatGPT coincided with significant changes in both the length and style of student essays, mirroring observations in other contexts such as academic publishing. We also observe -- as expected based on related public discourse -- changes in prevalence of key content words related to AI and LLMs, but not necessarily the general themes or topics discussed in the student essays as identified through (dynamic) topic modeling. 

---
# A Phenomenological Approach to Analyzing User Queries in IT Systems Using Heidegger's Fundamental Ontology 

**Authors**: Maksim Vishnevskiy  

**Link**: [PDF](https://arxiv.org/pdf/2504.12977)  

**Abstract**: This paper presents a novel research analytical IT system grounded in Martin Heidegger's Fundamental Ontology, distinguishing between beings (das Seiende) and Being (das Sein). The system employs two modally distinct, descriptively complete languages: a categorical language of beings for processing user inputs and an existential language of Being for internal analysis. These languages are bridged via a phenomenological reduction module, enabling the system to analyze user queries (including questions, answers, and dialogues among IT specialists), identify recursive and self-referential structures, and provide actionable insights in categorical terms. Unlike contemporary systems limited to categorical analysis, this approach leverages Heidegger's phenomenological existential analysis to uncover deeper ontological patterns in query processing, aiding in resolving logical traps in complex interactions, such as metaphor usage in IT contexts. The path to full realization involves formalizing the language of Being by a research team based on Heidegger's Fundamental Ontology; given the existing completeness of the language of beings, this reduces the system's computability to completeness, paving the way for a universal query analysis tool. The paper presents the system's architecture, operational principles, technical implementation, use cases--including a case based on real IT specialist dialogues--comparative evaluation with existing tools, and its advantages and limitations. 

---
# Building Russian Benchmark for Evaluation of Information Retrieval Models 

**Authors**: Grigory Kovalev, Mikhail Tikhomirov, Evgeny Kozhevnikov, Max Kornilov, Natalia Loukachevitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.12879)  

**Abstract**: We introduce RusBEIR, a comprehensive benchmark designed for zero-shot evaluation of information retrieval (IR) models in the Russian language. Comprising 17 datasets from various domains, it integrates adapted, translated, and newly created datasets, enabling systematic comparison of lexical and neural models. Our study highlights the importance of preprocessing for lexical models in morphologically rich languages and confirms BM25 as a strong baseline for full-document retrieval. Neural models, such as mE5-large and BGE-M3, demonstrate superior performance on most datasets, but face challenges with long-document retrieval due to input size constraints. RusBEIR offers a unified, open-source framework that promotes research in Russian-language information retrieval. 

---
# EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting 

**Authors**: Guanrou Yang, Chen Yang, Qian Chen, Ziyang Ma, Wenxi Chen, Wen Wang, Tianrui Wang, Yifan Yang, Zhikang Niu, Wenrui Liu, Fan Yu, Zhihao Du, Zhifu Gao, ShiLiang Zhang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12867)  

**Abstract**: Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and modality-of-thought (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at this https URL. Dataset, code, and checkpoints will be released. 

---
# Towards Lossless Token Pruning in Late-Interaction Retrieval Models 

**Authors**: Yuxuan Zong, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2504.12778)  

**Abstract**: Late interaction neural IR models like ColBERT offer a competitive effectiveness-efficiency trade-off across many benchmarks. However, they require a huge memory space to store the contextual representation for all the document tokens. Some works have proposed using either heuristics or statistical-based techniques to prune tokens from each document. This however doesn't guarantee that the removed tokens have no impact on the retrieval score. Our work uses a principled approach to define how to prune tokens without impacting the score between a document and a query. We introduce three regularization losses, that induce a solution with high pruning ratios, as well as two pruning strategies. We study them experimentally (in and out-domain), showing that we can preserve ColBERT's performance while using only 30\% of the tokens. 

---
# WebLists: Extracting Structured Information From Complex Interactive Websites Using Executable LLM Agents 

**Authors**: Arth Bohra, Manvel Saroyan, Danil Melkozerov, Vahe Karufanyan, Gabriel Maher, Pascal Weinberger, Artem Harutyunyan, Giovanni Campagna  

**Link**: [PDF](https://arxiv.org/pdf/2504.12682)  

**Abstract**: Most recent web agent research has focused on navigation and transaction tasks, with little emphasis on extracting structured data at scale. We present WebLists, a benchmark of 200 data-extraction tasks across four common business and enterprise use-cases. Each task requires an agent to navigate to a webpage, configure it appropriately, and extract complete datasets with well-defined schemas. We show that both LLMs with search capabilities and SOTA web agents struggle with these tasks, with a recall of 3% and 31%, respectively, despite higher performance on question-answering tasks.
To address this challenge, we propose BardeenAgent, a novel framework that enables web agents to convert their execution into repeatable programs, and replay them at scale across pages with similar structure. BardeenAgent is also the first LLM agent to take advantage of the regular structure of HTML. In particular BardeenAgent constructs a generalizable CSS selector to capture all relevant items on the page, then fits the operations to extract the data.
On the WebLists benchmark, BardeenAgent achieves 66% recall overall, more than doubling the performance of SOTA web agents, and reducing cost per output row by 3x. 

---
# VLMGuard-R1: Proactive Safety Alignment for VLMs via Reasoning-Driven Prompt Optimization 

**Authors**: Menglan Chen, Xianghe Pang, Jingjing Dong, WenHao Wang, Yaxin Du, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12661)  

**Abstract**: Aligning Vision-Language Models (VLMs) with safety standards is essential to mitigate risks arising from their multimodal complexity, where integrating vision and language unveils subtle threats beyond the reach of conventional safeguards. Inspired by the insight that reasoning across modalities is key to preempting intricate vulnerabilities, we propose a novel direction for VLM safety: multimodal reasoning-driven prompt rewriting. To this end, we introduce VLMGuard-R1, a proactive framework that refines user inputs through a reasoning-guided rewriter, dynamically interpreting text-image interactions to deliver refined prompts that bolster safety across diverse VLM architectures without altering their core parameters. To achieve this, we devise a three-stage reasoning pipeline to synthesize a dataset that trains the rewriter to infer subtle threats, enabling tailored, actionable responses over generic refusals. Extensive experiments across three benchmarks with five VLMs reveal that VLMGuard-R1 outperforms four baselines. In particular, VLMGuard-R1 achieves a remarkable 43.59\% increase in average safety across five models on the SIUO benchmark. 

---
# Simplifying Graph Transformers 

**Authors**: Liheng Ma, Soumyasundar Pal, Yingxue Zhang, Philip H.S. Torr, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2504.12588)  

**Abstract**: Transformers have attained outstanding performance across various modalities, employing scaled-dot-product (SDP) attention mechanisms. Researchers have attempted to migrate Transformers to graph learning, but most advanced Graph Transformers are designed with major architectural differences, either integrating message-passing or incorporating sophisticated attention mechanisms. These complexities prevent the easy adoption of Transformer training advances. We propose three simple modifications to the plain Transformer to render it applicable to graphs without introducing major architectural distortions. Specifically, we advocate for the use of (1) simplified $L_2$ attention to measure the magnitude closeness of tokens; (2) adaptive root-mean-square normalization to preserve token magnitude information; and (3) a relative positional encoding bias with a shared encoder. Significant performance gains across a variety of graph datasets justify the effectiveness of our proposed modifications. Furthermore, empirical evaluation on the expressiveness benchmark reveals noteworthy realized expressiveness in the graph isomorphism. 

---
# Provable Secure Steganography Based on Adaptive Dynamic Sampling 

**Authors**: Kaiyi Pang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12579)  

**Abstract**: The security of private communication is increasingly at risk due to widespread surveillance. Steganography, a technique for embedding secret messages within innocuous carriers, enables covert communication over monitored channels. Provably Secure Steganography (PSS) is state of the art for making stego carriers indistinguishable from normal ones by ensuring computational indistinguishability between stego and cover distributions. However, current PSS methods often require explicit access to the distribution of generative model for both sender and receiver, limiting their practicality in black box scenarios. In this paper, we propose a provably secure steganography scheme that does not require access to explicit model distributions for both sender and receiver. Our method incorporates a dynamic sampling strategy, enabling generative models to embed secret messages within multiple sampling choices without disrupting the normal generation process of the model. Extensive evaluations of three real world datasets and three LLMs demonstrate that our blackbox method is comparable with existing white-box steganography methods in terms of efficiency and capacity while eliminating the degradation of steganography in model generated outputs. 

---
# ZeroSumEval: Scaling LLM Evaluation with Inter-Model Competition 

**Authors**: Haidar Khan, Hisham A. Alyahya, Yazeed Alnumay, M Saiful Bari, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2504.12562)  

**Abstract**: Evaluating the capabilities of Large Language Models (LLMs) has traditionally relied on static benchmark datasets, human assessments, or model-based evaluations - methods that often suffer from overfitting, high costs, and biases. ZeroSumEval is a novel competition-based evaluation protocol that leverages zero-sum games to assess LLMs with dynamic benchmarks that resist saturation. ZeroSumEval encompasses a diverse suite of games, including security challenges (PyJail), classic games (Chess, Liar's Dice, Poker), knowledge tests (MathQuiz), and persuasion challenges (Gandalf, Debate). These games are designed to evaluate a range of AI capabilities such as strategic reasoning, planning, knowledge application, and creativity. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework. To demonstrate this, we conduct extensive experiments with >7000 simulations across 7 games and 13 models. Our results show that while frontier models from the GPT and Claude families can play common games and answer questions, they struggle to play games that require creating novel and challenging questions. We also observe that models cannot reliably jailbreak each other and fail generally at tasks requiring creativity. We release our code at this https URL. 

---
# Benchmarking LLM-based Relevance Judgment Methods 

**Authors**: Negar Arabzadeh, Charles L. A. Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2504.12558)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in both academic and industry settings to automate the evaluation of information seeking systems, particularly by generating graded relevance judgments. Previous work on LLM-based relevance assessment has primarily focused on replicating graded human relevance judgments through various prompting strategies. However, there has been limited exploration of alternative assessment methods or comprehensive comparative studies. In this paper, we systematically compare multiple LLM-based relevance assessment methods, including binary relevance judgments, graded relevance assessments, pairwise preference-based methods, and two nugget-based evaluation methods~--~document-agnostic and document-dependent. In addition to a traditional comparison based on system rankings using Kendall correlations, we also examine how well LLM judgments align with human preferences, as inferred from relevance grades. We conduct extensive experiments on datasets from three TREC Deep Learning tracks 2019, 2020 and 2021 as well as the ANTIQUE dataset, which focuses on non-factoid open-domain question answering. As part of our data release, we include relevance judgments generated by both an open-source (Llama3.2b) and a commercial (gpt-4o) model. Our goal is to \textit{reproduce} various LLM-based relevance judgment methods to provide a comprehensive comparison. All code, data, and resources are publicly available in our GitHub Repository at this https URL. 

---
# Knowledge Acquisition on Mass-shooting Events via LLMs for AI-Driven Justice 

**Authors**: Benign John Ihugba, Afsana Nasrin, Ling Wu, Lin Li, Lijun Qian, Xishuang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.12545)  

**Abstract**: Mass-shooting events pose a significant challenge to public safety, generating large volumes of unstructured textual data that hinder effective investigations and the formulation of public policy. Despite the urgency, few prior studies have effectively automated the extraction of key information from these events to support legal and investigative efforts. This paper presented the first dataset designed for knowledge acquisition on mass-shooting events through the application of named entity recognition (NER) techniques. It focuses on identifying key entities such as offenders, victims, locations, and criminal instruments, that are vital for legal and investigative purposes. The NER process is powered by Large Language Models (LLMs) using few-shot prompting, facilitating the efficient extraction and organization of critical information from diverse sources, including news articles, police reports, and social media. Experimental results on real-world mass-shooting corpora demonstrate that GPT-4o is the most effective model for mass-shooting NER, achieving the highest Micro Precision, Micro Recall, and Micro F1-scores. Meanwhile, o1-mini delivers competitive performance, making it a resource-efficient alternative for less complex NER tasks. It is also observed that increasing the shot count enhances the performance of all models, but the gains are more substantial for GPT-4o and o1-mini, highlighting their superior adaptability to few-shot learning scenarios. 

---
# Towards Conversational AI for Human-Machine Collaborative MLOps 

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Anargyros Tsadimas, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.12477)  

**Abstract**: This paper presents a Large Language Model (LLM) based conversational agent system designed to enhance human-machine collaboration in Machine Learning Operations (MLOps). We introduce the Swarm Agent, an extensible architecture that integrates specialized agents to create and manage ML workflows through natural language interactions. The system leverages a hierarchical, modular design incorporating a KubeFlow Pipelines (KFP) Agent for ML pipeline orchestration, a MinIO Agent for data management, and a Retrieval-Augmented Generation (RAG) Agent for domain-specific knowledge integration. Through iterative reasoning loops and context-aware processing, the system enables users with varying technical backgrounds to discover, execute, and monitor ML pipelines; manage datasets and artifacts; and access relevant documentation, all via intuitive conversational interfaces. Our approach addresses the accessibility gap in complex MLOps platforms like Kubeflow, making advanced ML tools broadly accessible while maintaining the flexibility to extend to other platforms. The paper describes the architecture, implementation details, and demonstrates how this conversational MLOps assistant reduces complexity and lowers barriers to entry for users across diverse technical skill levels. 

---
# A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment 

**Authors**: Negar Arabzadeh, Charles L. A . Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2504.12408)  

**Abstract**: Large Language Models (LLMs) are increasingly used to automate relevance judgments for information retrieval (IR) tasks, often demonstrating agreement with human labels that approaches inter-human agreement. To assess the robustness and reliability of LLM-based relevance judgments, we systematically investigate impact of prompt sensitivity on the task. We collected prompts for relevance assessment from 15 human experts and 15 LLMs across three tasks~ -- ~binary, graded, and pairwise~ -- ~yielding 90 prompts in total. After filtering out unusable prompts from three humans and three LLMs, we employed the remaining 72 prompts with three different LLMs as judges to label document/query pairs from two TREC Deep Learning Datasets (2020 and 2021). We compare LLM-generated labels with TREC official human labels using Cohen's $\kappa$ and pairwise agreement measures. In addition to investigating the impact of prompt variations on agreement with human labels, we compare human- and LLM-generated prompts and analyze differences among different LLMs as judges. We also compare human- and LLM-generated prompts with the standard UMBRELA prompt used for relevance assessment by Bing and TREC 2024 Retrieval Augmented Generation (RAG) Track. To support future research in LLM-based evaluation, we release all data and prompts at this https URL. 

---
# Specialized text classification: an approach to classifying Open Banking transactions 

**Authors**: Duc Tuyen TA, Wajdi Ben Saad, Ji Young Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.12319)  

**Abstract**: With the introduction of the PSD2 regulation in the EU which established the Open Banking framework, a new window of opportunities has opened for banks and fintechs to explore and enrich Bank transaction descriptions with the aim of building a better understanding of customer behavior, while using this understanding to prevent fraud, reduce risks and offer more competitive and tailored services.
And although the usage of natural language processing models and techniques has seen an incredible progress in various applications and domains over the past few years, custom applications based on domain-specific text corpus remain unaddressed especially in the banking sector.
In this paper, we introduce a language-based Open Banking transaction classification system with a focus on the french market and french language text. The system encompasses data collection, labeling, preprocessing, modeling, and evaluation stages. Unlike previous studies that focus on general classification approaches, this system is specifically tailored to address the challenges posed by training a language model with a specialized text corpus (Banking data in the French context). By incorporating language-specific techniques and domain knowledge, the proposed system demonstrates enhanced performance and efficiency compared to generic approaches. 

---
# Large Language Model-Based Knowledge Graph System Construction for Sustainable Development Goals: An AI-Based Speculative Design Perspective 

**Authors**: Yi-De Lin, Guan-Ze Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12309)  

**Abstract**: From 2000 to 2015, the UN's Millennium Development Goals guided global priorities. The subsequent Sustainable Development Goals (SDGs) adopted a more dynamic approach, with annual indicator updates. As 2030 nears and progress lags, innovative acceleration strategies are critical. This study develops an AI-powered knowledge graph system to analyze SDG interconnections, discover potential new goals, and visualize them online. Using official SDG texts, Elsevier's keyword dataset, and 1,127 TED Talk transcripts (2020-2023), a pilot on 269 talks from 2023 applies AI-speculative design, large language models, and retrieval-augmented generation. Key findings include: (1) Heatmap analysis reveals strong associations between Goal 10 and Goal 16, and minimal coverage of Goal 6. (2) In the knowledge graph, simulated dialogue over time reveals new central nodes, showing how richer data supports divergent thinking and goal clarity. (3) Six potential new goals are proposed, centered on equity, resilience, and technology-driven inclusion. This speculative-AI framework offers fresh insights for policymakers and lays groundwork for future multimodal and cross-system SDG applications. 

---
