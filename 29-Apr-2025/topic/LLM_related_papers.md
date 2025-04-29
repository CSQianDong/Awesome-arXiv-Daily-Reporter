# GenCLS++: Pushing the Boundaries of Generative Classification in LLMs Through Comprehensive SFT and RL Studies Across Diverse Datasets 

**Authors**: Mingqian He, Fei Zhao, Chonggang Lu, Ziyan Liu, Yue Wang, Haofu Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.19898)  

**Abstract**: As a fundamental task in machine learning, text classification plays a crucial role in many areas. With the rapid scaling of Large Language Models (LLMs), particularly through reinforcement learning (RL), there is a growing need for more capable discriminators. Consequently, advances in classification are becoming increasingly vital for enhancing the overall capabilities of LLMs. Traditional discriminative methods map text to labels but overlook LLMs' intrinsic generative strengths. Generative classification addresses this by prompting the model to directly output labels. However, existing studies still rely on simple SFT alone, seldom probing the interplay between training and inference prompts, and no work has systematically leveraged RL for generative text classifiers and unified SFT, RL, and inference-time prompting in one framework. We bridge this gap with GenCLS++, a framework that jointly optimizes SFT and RL while systematically exploring five high-level strategy dimensions-in-context learning variants, category definitions, explicit uncertainty labels, semantically irrelevant numeric labels, and perplexity-based decoding-during both training and inference. After an SFT "policy warm-up," we apply RL with a simple rule-based reward, yielding sizable extra gains. Across seven datasets, GenCLS++ achieves an average accuracy improvement of 3.46% relative to the naive SFT baseline; on public datasets, this improvement rises to 4.00%. Notably, unlike reasoning-intensive tasks that benefit from explicit thinking processes, we find that classification tasks perform better without such reasoning steps. These insights into the role of explicit reasoning provide valuable guidance for future LLM applications. 

---
# Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages 

**Authors**: Pritika Rohera, Chaitrali Ginimav, Gayatri Sawant, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20022)  

**Abstract**: Multilingual Large Language Models (LLMs) have demonstrated significant effectiveness across various languages, particularly in high-resource languages such as English. However, their performance in terms of factual accuracy across other low-resource languages, especially Indic languages, remains an area of investigation. In this study, we assess the factual accuracy of LLMs - GPT-4o, Gemma-2-9B, Gemma-2-2B, and Llama-3.1-8B - by comparing their performance in English and Indic languages using the IndicQuest dataset, which contains question-answer pairs in English and 19 Indic languages. By asking the same questions in English and their respective Indic translations, we analyze whether the models are more reliable for regional context questions in Indic languages or when operating in English. Our findings reveal that LLMs often perform better in English, even for questions rooted in Indic contexts. Notably, we observe a higher tendency for hallucination in responses generated in low-resource Indic languages, highlighting challenges in the multilingual understanding capabilities of current LLMs. 

---
# Moral Reasoning Across Languages: The Critical Role of Low-Resource Languages in LLMs 

**Authors**: Huichi Zhou, Zehao Xu, Munan Zhao, Kaihong Li, Yiqiang Li, Hongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19759)  

**Abstract**: In this paper, we introduce the Multilingual Moral Reasoning Benchmark (MMRB) to evaluate the moral reasoning abilities of large language models (LLMs) across five typologically diverse languages and three levels of contextual complexity: sentence, paragraph, and document. Our results show moral reasoning performance degrades with increasing context complexity, particularly for low-resource languages such as Vietnamese. We further fine-tune the open-source LLaMA-3-8B model using curated monolingual data for alignment and poisoning. Surprisingly, low-resource languages have a stronger impact on multilingual reasoning than high-resource ones, highlighting their critical role in multilingual NLP. 

---
# Annif at SemEval-2025 Task 5: Traditional XMTC augmented by LLMs 

**Authors**: Osma Suominen, Juho Inkinen, Mona Lehtinen  

**Link**: [PDF](https://arxiv.org/pdf/2504.19675)  

**Abstract**: This paper presents the Annif system in SemEval-2025 Task 5 (LLMs4Subjects), which focussed on subject indexing using large language models (LLMs). The task required creating subject predictions for bibliographic records from the bilingual TIBKAT database using the GND subject vocabulary. Our approach combines traditional natural language processing and machine learning techniques implemented in the Annif toolkit with innovative LLM-based methods for translation and synthetic data generation, and merging predictions from monolingual models. The system ranked first in the all-subjects category and second in the tib-core-subjects category in the quantitative evaluation, and fourth in qualitative evaluations. These findings demonstrate the potential of combining traditional XMTC algorithms with modern LLM techniques to improve the accuracy and efficiency of subject indexing in multilingual contexts. 

---
# LLM-Assisted Automated Deductive Coding of Dialogue Data: Leveraging Dialogue-Specific Characteristics to Enhance Contextual Understanding 

**Authors**: Ying Na, Shihui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.19734)  

**Abstract**: Dialogue data has been a key source for understanding learning processes, offering critical insights into how students engage in collaborative discussions and how these interactions shape their knowledge construction. The advent of Large Language Models (LLMs) has introduced promising opportunities for advancing qualitative research, particularly in the automated coding of dialogue data. However, the inherent contextual complexity of dialogue presents unique challenges for these models, especially in understanding and interpreting complex contextual information. This study addresses these challenges by developing a novel LLM-assisted automated coding approach for dialogue data. The novelty of our proposed framework is threefold: 1) We predict the code for an utterance based on dialogue-specific characteristics -- communicative acts and communicative events -- using separate prompts following the role prompts and chain-of-thoughts methods; 2) We engaged multiple LLMs including GPT-4-turbo, GPT-4o, DeepSeek in collaborative code prediction; 3) We leveraged the interrelation between events and acts to implement consistency checking using GPT-4o. In particular, our contextual consistency checking provided a substantial accuracy improvement. We also found the accuracy of act predictions was consistently higher than that of event predictions. This study contributes a new methodological framework for enhancing the precision of automated coding of dialogue data as well as offers a scalable solution for addressing the contextual challenges inherent in dialogue analysis. 

---
# m-KAILIN: Knowledge-Driven Agentic Scientific Corpus Distillation Framework for Biomedical Large Language Models Training 

**Authors**: Meng Xiao, Xunxin Cai, Chengrui Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.19565)  

**Abstract**: The rapid progress of large language models (LLMs) in biomedical research has underscored the limitations of existing open-source annotated scientific corpora, which are often insufficient in quantity and quality. Addressing the challenge posed by the complex hierarchy of biomedical knowledge, we propose a knowledge-driven, multi-agent framework for scientific corpus distillation tailored for LLM training in the biomedical domain. Central to our approach is a collaborative multi-agent architecture, where specialized agents, each guided by the Medical Subject Headings (MeSH) hierarchy, work in concert to autonomously extract, synthesize, and self-evaluate high-quality textual data from vast scientific literature. These agents collectively generate and refine domain-specific question-answer pairs, ensuring comprehensive coverage and consistency with biomedical ontologies while minimizing manual involvement. Extensive experimental results show that language models trained on our multi-agent distilled datasets achieve notable improvements in biomedical question-answering tasks, outperforming both strong life sciences LLM baselines and advanced proprietary models. Notably, our AI-Ready dataset enables Llama3-70B to surpass GPT-4 with MedPrompt and Med-PaLM-2, despite their larger scale. Detailed ablation studies and case analyses further validate the effectiveness and synergy of each agent within the framework, highlighting the potential of multi-agent collaboration in biomedical LLM training. 

---
# BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text 

**Authors**: Jiageng Wu, Bowen Gu, Ren Zhou, Kevin Xie, Doug Snyder, Yixing Jiang, Valentina Carducci, Richard Wyss, Rishi J Desai, Emily Alsentzer, Leo Anthony Celi, Adam Rodman, Sebastian Schneeweiss, Jonathan H. Chen, Santiago Romero-Brufau, Kueiyu Joshua Lin, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19467)  

**Abstract**: Large language models (LLMs) hold great promise for medical applications and are evolving rapidly, with new models being released at an accelerated pace. However, current evaluations of LLMs in clinical contexts remain limited. Most existing benchmarks rely on medical exam-style questions or PubMed-derived text, failing to capture the complexity of real-world electronic health record (EHR) data. Others focus narrowly on specific application scenarios, limiting their generalizability across broader clinical use. To address this gap, we present BRIDGE, a comprehensive multilingual benchmark comprising 87 tasks sourced from real-world clinical data sources across nine languages. We systematically evaluated 52 state-of-the-art LLMs (including DeepSeek-R1, GPT-4o, Gemini, and Llama 4) under various inference strategies. With a total of 13,572 experiments, our results reveal substantial performance variation across model sizes, languages, natural language processing tasks, and clinical specialties. Notably, we demonstrate that open-source LLMs can achieve performance comparable to proprietary models, while medically fine-tuned LLMs based on older architectures often underperform versus updated general-purpose models. The BRIDGE and its corresponding leaderboard serve as a foundational resource and a unique reference for the development and evaluation of new LLMs in real-world clinical text understanding. 

---
# LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation 

**Authors**: Beizhe Hu, Qiang Sheng, Juan Cao, Yang Li, Danding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20013)  

**Abstract**: Online fake news moderation now faces a new challenge brought by the malicious use of large language models (LLMs) in fake news production. Though existing works have shown LLM-generated fake news is hard to detect from an individual aspect, it remains underexplored how its large-scale release will impact the news ecosystem. In this study, we develop a simulation pipeline and a dataset with ~56k generated news of diverse types to investigate the effects of LLM-generated fake news within neural news recommendation systems. Our findings expose a truth decay phenomenon, where real news is gradually losing its advantageous position in news ranking against fake news as LLM-generated news is involved in news recommendation. We further provide an explanation about why truth decay occurs from a familiarity perspective and show the positive correlation between perplexity and news ranking. Finally, we discuss the threats of LLM-generated fake news and provide possible countermeasures. We urge stakeholders to address this emerging challenge to preserve the integrity of news ecosystems. 

---
# Assessing the Potential of Generative Agents in Crowdsourced Fact-Checking 

**Authors**: Luigia Costabile, Gian Marco Orlando, Valerio La Gatta, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2504.19940)  

**Abstract**: The growing spread of online misinformation has created an urgent need for scalable, reliable fact-checking solutions. Crowdsourced fact-checking - where non-experts evaluate claim veracity - offers a cost-effective alternative to expert verification, despite concerns about variability in quality and bias. Encouraged by promising results in certain contexts, major platforms such as X (formerly Twitter), Facebook, and Instagram have begun shifting from centralized moderation to decentralized, crowd-based approaches.
In parallel, advances in Large Language Models (LLMs) have shown strong performance across core fact-checking tasks, including claim detection and evidence evaluation. However, their potential role in crowdsourced workflows remains unexplored. This paper investigates whether LLM-powered generative agents - autonomous entities that emulate human behavior and decision-making - can meaningfully contribute to fact-checking tasks traditionally reserved for human crowds. Using the protocol of La Barbera et al. (2024), we simulate crowds of generative agents with diverse demographic and ideological profiles. Agents retrieve evidence, assess claims along multiple quality dimensions, and issue final veracity judgments.
Our results show that agent crowds outperform human crowds in truthfulness classification, exhibit higher internal consistency, and show reduced susceptibility to social and cognitive biases. Compared to humans, agents rely more systematically on informative criteria such as Accuracy, Precision, and Informativeness, suggesting a more structured decision-making process. Overall, our findings highlight the potential of generative agents as scalable, consistent, and less biased contributors to crowd-based fact-checking systems. 

---
# Coreference Resolution for Vietnamese Narrative Texts 

**Authors**: Hieu-Dai Tran, Duc-Vu Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.19606)  

**Abstract**: Coreference resolution is a vital task in natural language processing (NLP) that involves identifying and linking different expressions in a text that refer to the same entity. This task is particularly challenging for Vietnamese, a low-resource language with limited annotated datasets. To address these challenges, we developed a comprehensive annotated dataset using narrative texts from VnExpress, a widely-read Vietnamese online news platform. We established detailed guidelines for annotating entities, focusing on ensuring consistency and accuracy. Additionally, we evaluated the performance of large language models (LLMs), specifically GPT-3.5-Turbo and GPT-4, on this dataset. Our results demonstrate that GPT-4 significantly outperforms GPT-3.5-Turbo in terms of both accuracy and response consistency, making it a more reliable tool for coreference resolution in Vietnamese. 

---
# Systematic Bias in Large Language Models: Discrepant Response Patterns in Binary vs. Continuous Judgment Tasks 

**Authors**: Yi-Long Lu, Chunhui Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19445)  

**Abstract**: Large Language Models (LLMs) are increasingly used in tasks such as psychological text analysis and decision-making in automated workflows. However, their reliability remains a concern due to potential biases inherited from their training process. In this study, we examine how different response format: binary versus continuous, may systematically influence LLMs' judgments. In a value statement judgments task and a text sentiment analysis task, we prompted LLMs to simulate human responses and tested both formats across several models, including both open-source and commercial models. Our findings revealed a consistent negative bias: LLMs were more likely to deliver "negative" judgments in binary formats compared to continuous ones. Control experiments further revealed that this pattern holds across both tasks. Our results highlight the importance of considering response format when applying LLMs to decision tasks, as small changes in task design can introduce systematic biases. 

---
# AutoJudge: Judge Decoding Without Manual Annotation 

**Authors**: Roman Garipov, Fedor Velikonivtsev, Ruslan Svirschevski, Vage Egiazarian, Max Ryabinin  

**Link**: [PDF](https://arxiv.org/pdf/2504.20039)  

**Abstract**: We introduce AutoJudge, a framework that accelerates large language model (LLM) inference with task-specific lossy speculative decoding. Instead of matching the original model output distribution token-by-token, we identify which of the generated tokens affect the downstream quality of the generated response, relaxing the guarantee so that the "unimportant" tokens can be generated faster. Our approach relies on a semi-greedy search algorithm to test which of the mismatches between target and draft model should be corrected to preserve quality, and which ones may be skipped. We then train a lightweight classifier based on existing LLM embeddings to predict, at inference time, which mismatching tokens can be safely accepted without compromising the final answer quality. We test our approach with Llama 3.2 1B (draft) and Llama 3.1 8B (target) models on zero-shot GSM8K reasoning, where it achieves up to 1.5x more accepted tokens per verification cycle with under 1% degradation in answer accuracy compared to standard speculative decoding and over 2x with small loss in accuracy. When applied to the LiveCodeBench benchmark, our approach automatically detects other, programming-specific important tokens and shows similar speedups, demonstrating its ability to generalize across tasks. 

---
# Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory 

**Authors**: Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2504.19413)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable prowess in generating contextually coherent responses, yet their fixed context windows pose fundamental challenges for maintaining consistency over prolonged multi-session dialogues. We introduce Mem0, a scalable memory-centric architecture that addresses this issue by dynamically extracting, consolidating, and retrieving salient information from ongoing conversations. Building on this foundation, we further propose an enhanced variant that leverages graph-based memory representations to capture complex relational structures among conversational elements. Through comprehensive evaluations on LOCOMO benchmark, we systematically compare our approaches against six baseline categories: (i) established memory-augmented systems, (ii) retrieval-augmented generation (RAG) with varying chunk sizes and k-values, (iii) a full-context approach that processes the entire conversation history, (iv) an open-source memory solution, (v) a proprietary model system, and (vi) a dedicated memory management platform. Empirical results show that our methods consistently outperform all existing memory systems across four question categories: single-hop, temporal, multi-hop, and open-domain. Notably, Mem0 achieves 26% relative improvements in the LLM-as-a-Judge metric over OpenAI, while Mem0 with graph memory achieves around 2% higher overall score than the base configuration. Beyond accuracy gains, we also markedly reduce computational overhead compared to full-context method. In particular, Mem0 attains a 91% lower p95 latency and saves more than 90% token cost, offering a compelling balance between advanced reasoning capabilities and practical deployment constraints. Our findings highlight critical role of structured, persistent memory mechanisms for long-term conversational coherence, paving the way for more reliable and efficient LLM-driven AI agents. 

---
# Taming the Titans: A Survey of Efficient LLM Inference Serving 

**Authors**: Ranran Zhen, Juntao Li, Yixin Ji, Zhenlin Yang, Tong Liu, Qingrong Xia, Xinyu Duan, Zhefeng Wang, Baoxing Huai, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19720)  

**Abstract**: Large Language Models (LLMs) for Generative AI have achieved remarkable progress, evolving into sophisticated and versatile tools widely adopted across various domains and applications. However, the substantial memory overhead caused by their vast number of parameters, combined with the high computational demands of the attention mechanism, poses significant challenges in achieving low latency and high throughput for LLM inference services. Recent advancements, driven by groundbreaking research, have significantly accelerated progress in this field. This paper provides a comprehensive survey of these methods, covering fundamental instance-level approaches, in-depth cluster-level strategies, emerging scenario directions, and other miscellaneous but important areas. At the instance level, we review model placement, request scheduling, decoding length prediction, storage management, and the disaggregation paradigm. At the cluster level, we explore GPU cluster deployment, multi-instance load balancing, and cloud service solutions. For emerging scenarios, we organize the discussion around specific tasks, modules, and auxiliary methods. To ensure a holistic overview, we also highlight several niche yet critical areas. Finally, we outline potential research directions to further advance the field of LLM inference serving. 

---
# BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese 

**Authors**: Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, Yuxin Gu, Sixin Hong, Jing Ren, Jian Chen, Chao Liu, Yining Hua  

**Link**: [PDF](https://arxiv.org/pdf/2504.19314)  

**Abstract**: As large language models (LLMs) evolve into tool-using agents, the ability to browse the web in real-time has become a critical yardstick for measuring their reasoning and retrieval competence. Existing benchmarks such as BrowseComp concentrate on English and overlook the linguistic, infrastructural, and censorship-related complexities of other major information ecosystems -- most notably Chinese. To address this gap, we introduce BrowseComp-ZH, a high-difficulty benchmark purpose-built to comprehensively evaluate LLM agents on the Chinese web. BrowseComp-ZH consists of 289 multi-hop questions spanning 11 diverse domains. Each question is reverse-engineered from a short, objective, and easily verifiable answer (e.g., a date, number, or proper noun). A two-stage quality control protocol is applied to strive for high question difficulty and answer uniqueness. We benchmark over 20 state-of-the-art language models and agentic search systems on our proposed BrowseComp-ZH. Despite their strong conversational and retrieval capabilities, most models struggle severely: a large number achieve accuracy rates below 10%, and only a handful exceed 20%. Even the best-performing system, OpenAI's DeepResearch, reaches just 42.9%. These results demonstrate the considerable difficulty of BrowseComp-ZH, where success demands not only effective retrieval strategies, but also sophisticated reasoning and information reconciliation -- capabilities that current models still struggle to master. Our dataset, construction guidelines, and benchmark results have been publicly released at this https URL. 

---
# Towards Long Context Hallucination Detection 

**Authors**: Siyi Liu, Kishaloy Halder, Zheng Qi, Wei Xiao, Nikolaos Pappas, Phu Mon Htut, Neha Anna John, Yassine Benajiba, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2504.19457)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various tasks. However, they are prone to contextual hallucination, generating information that is either unsubstantiated or contradictory to the given context. Although many studies have investigated contextual hallucinations in LLMs, addressing them in long-context inputs remains an open problem. In this work, we take an initial step toward solving this problem by constructing a dataset specifically designed for long-context hallucination detection. Furthermore, we propose a novel architecture that enables pre-trained encoder models, such as BERT, to process long contexts and effectively detect contextual hallucinations through a decomposition and aggregation mechanism. Our experimental results show that the proposed architecture significantly outperforms previous models of similar size as well as LLM-based models across various metrics, while providing substantially faster inference. 

---
# Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers 

**Authors**: Dylan Bouchard, Mohit Singh Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.19254)  

**Abstract**: Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we propose a versatile framework for zero-resource hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we introduce a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs. 

---
# SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning 

**Authors**: Jiaqi Chen, Bang Zhang, Ruotian Ma, Peisong Wang, Xiaodan Liang, Zhaopeng Tu, Xiaolong Li, Kwan-Yee K. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2504.19162)  

**Abstract**: Evaluating the step-by-step reliability of large language model (LLM) reasoning, such as Chain-of-Thought, remains challenging due to the difficulty and cost of obtaining high-quality step-level supervision. In this paper, we introduce Self-Play Critic (SPC), a novel approach where a critic model evolves its ability to assess reasoning steps through adversarial self-play games, eliminating the need for manual step-level annotation. SPC involves fine-tuning two copies of a base model to play two roles, namely a "sneaky generator" that deliberately produces erroneous steps designed to be difficult to detect, and a "critic" that analyzes the correctness of reasoning steps. These two models engage in an adversarial game in which the generator aims to fool the critic, while the critic model seeks to identify the generator's errors. Using reinforcement learning based on the game outcomes, the models iteratively improve; the winner of each confrontation receives a positive reward and the loser receives a negative reward, driving continuous self-evolution. Experiments on three reasoning process benchmarks (ProcessBench, PRM800K, DeltaBench) demonstrate that our SPC progressively enhances its error detection capabilities (e.g., accuracy increases from 70.8% to 77.7% on ProcessBench) and surpasses strong baselines, including distilled R1 model. Furthermore, applying SPC to guide the test-time search of diverse LLMs significantly improves their mathematical reasoning performance on MATH500 and AIME2024, outperforming state-of-the-art process reward models. 

---
# Context Selection and Rewriting for Video-based EducationalQuestion Generation 

**Authors**: Mengxia Yu, Bang Nguyen, Olivia Zino, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19406)  

**Abstract**: Educational question generation (EQG) is a crucial component of intelligent educational systems, significantly aiding self-assessment, active learning, and personalized education. While EQG systems have emerged, existing datasets typically rely on predefined, carefully edited texts, failing to represent real-world classroom content, including lecture speech with a set of complementary slides. To bridge this gap, we collect a dataset of educational questions based on lectures from real-world classrooms. On this realistic dataset, we find that current methods for EQG struggle with accurately generating questions from educational videos, particularly in aligning with specific timestamps and target answers. Common challenges include selecting informative contexts from extensive transcripts and ensuring generated questions meaningfully incorporate the target answer. To address the challenges, we introduce a novel framework utilizing large language models for dynamically selecting and rewriting contexts based on target timestamps and answers. First, our framework selects contexts from both lecture transcripts and video keyframes based on answer relevance and temporal proximity. Then, we integrate the contexts selected from both modalities and rewrite them into answer-containing knowledge statements, to enhance the logical connection between the contexts and the desired answer. This approach significantly improves the quality and relevance of the generated questions. Our dataset and code are released in this https URL. 

---
# APE-Bench I: Towards File-level Automated Proof Engineering of Formal Math Libraries 

**Authors**: Huajian Xin, Luming Li, Xiaoran Jin, Jacques Fleuriot, Wenda Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.19110)  

**Abstract**: Recent progress in large language models (LLMs) has shown promise in formal theorem proving, yet existing benchmarks remain limited to isolated, static proof tasks, failing to capture the iterative, engineering-intensive workflows of real-world formal mathematics libraries. Motivated by analogous advances in software engineering, we introduce the paradigm of Automated Proof Engineering (APE), which aims to automate proof engineering tasks such as feature addition, proof refactoring, and bug fixing using LLMs. To facilitate research in this direction, we present APE-Bench I, the first realistic benchmark built from real-world commit histories of Mathlib4, featuring diverse file-level tasks described in natural language and verified via a hybrid approach combining the Lean compiler and LLM-as-a-Judge. We further develop Eleanstic, a scalable parallel verification infrastructure optimized for proof checking across multiple versions of Mathlib. Empirical results on state-of-the-art LLMs demonstrate strong performance on localized edits but substantial degradation on handling complex proof engineering. This work lays the foundation for developing agentic workflows in proof engineering, with future benchmarks targeting multi-file coordination, project-scale verification, and autonomous agents capable of planning, editing, and repairing formal libraries. 

---
# Privacy-Preserving Federated Embedding Learning for Localized Retrieval-Augmented Generation 

**Authors**: Qianren Mao, Qili Zhang, Hanwen Hao, Zhentao Han, Runhua Xu, Weifeng Jiang, Qi Hu, Zhijun Chen, Tyler Zhou, Bo Li, Yangqiu Song, Jin Dong, Jianxin Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19101)  

**Abstract**: Retrieval-Augmented Generation (RAG) has recently emerged as a promising solution for enhancing the accuracy and credibility of Large Language Models (LLMs), particularly in Question & Answer tasks. This is achieved by incorporating proprietary and private data from integrated databases. However, private RAG systems face significant challenges due to the scarcity of private domain data and critical data privacy issues. These obstacles impede the deployment of private RAG systems, as developing privacy-preserving RAG systems requires a delicate balance between data security and data availability. To address these challenges, we regard federated learning (FL) as a highly promising technology for privacy-preserving RAG services. We propose a novel framework called Federated Retrieval-Augmented Generation (FedE4RAG). This framework facilitates collaborative training of client-side RAG retrieval models. The parameters of these models are aggregated and distributed on a central-server, ensuring data privacy without direct sharing of raw data. In FedE4RAG, knowledge distillation is employed for communication between the server and client models. This technique improves the generalization of local RAG retrievers during the federated learning process. Additionally, we apply homomorphic encryption within federated learning to safeguard model parameters and mitigate concerns related to data leakage. Extensive experiments conducted on the real-world dataset have validated the effectiveness of FedE4RAG. The results demonstrate that our proposed framework can markedly enhance the performance of private RAG systems while maintaining robust data privacy protection. 

---
# Calibrating Translation Decoding with Quality Estimation on LLMs 

**Authors**: Di Wu, Yibin Lei, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2504.19044)  

**Abstract**: Neural machine translation (NMT) systems typically employ maximum a posteriori (MAP) decoding to select the highest-scoring translation from the distribution mass. However, recent evidence highlights the inadequacy of MAP decoding, often resulting in low-quality or even pathological hypotheses -- the decoding objective is not aligned with real-world translation quality. This paper proposes calibrating hypothesis likelihoods with translation quality from a distribution view by directly optimizing their Pearson correlation -- thereby enhancing the effectiveness of translation decoding. With our method, translation on large language models (LLMs) improves substantially after limited training (2K instances per direction). This improvement is orthogonal to those achieved through supervised fine-tuning, leading to substantial gains across a broad range of metrics and human evaluations -- even when applied to top-performing translation-specialized LLMs fine-tuned on high-quality translation data, such as Tower, or when compared to recent preference optimization methods, like CPO. Moreover, the calibrated translation likelihood can directly serve as a strong proxy for translation quality, closely approximating or even surpassing some state-of-the-art translation quality estimation models, like CometKiwi. Lastly, our in-depth analysis demonstrates that calibration enhances the effectiveness of MAP decoding, thereby enabling greater efficiency in real-world deployment. The resulting state-of-the-art translation model, which covers 10 languages, along with the accompanying code and human evaluation data, has been released to the community: this https URL. 

---
# Hallucinations and Key Information Extraction in Medical Texts: A Comprehensive Assessment of Open-Source Large Language Models 

**Authors**: Anindya Bijoy Das, Shibbir Ahmed, Shahnewaz Karim Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2504.19061)  

**Abstract**: Clinical summarization is crucial in healthcare as it distills complex medical data into digestible information, enhancing patient understanding and care management. Large language models (LLMs) have shown significant potential in automating and improving the accuracy of such summarizations due to their advanced natural language understanding capabilities. These models are particularly applicable in the context of summarizing medical/clinical texts, where precise and concise information transfer is essential. In this paper, we investigate the effectiveness of open-source LLMs in extracting key events from discharge reports, such as reasons for hospital admission, significant in-hospital events, and critical follow-up actions. In addition, we also assess the prevalence of various types of hallucinations in the summaries produced by these models. Detecting hallucinations is vital as it directly influences the reliability of the information, potentially affecting patient care and treatment outcomes. We conduct comprehensive numerical simulations to rigorously evaluate the performance of these models, further probing the accuracy and fidelity of the extracted content in clinical summarization. 

---
# Efficient Reasoning for LLMs through Speculative Chain-of-Thought 

**Authors**: Jikai Wang, Juntao Li, Lijun Wu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19095)  

**Abstract**: Large reasoning language models such as OpenAI-o1 and Deepseek-R1 have recently attracted widespread attention due to their impressive task-solving abilities. However, the enormous model size and the generation of lengthy thought chains introduce significant reasoning costs and response latency. Existing methods for efficient reasoning mainly focus on reducing the number of model parameters or shortening the chain-of-thought length. In this paper, we introduce Speculative Chain-of-Thought (SCoT), which reduces reasoning latency from another perspective by accelerated average reasoning speed through large and small model collaboration. SCoT conducts thought-level drafting using a lightweight draft model. Then it selects the best CoT draft and corrects the error cases with the target model. The proposed thinking behavior alignment improves the efficiency of drafting and the draft selection strategy maintains the prediction accuracy for complex problems. Experimental results on GSM8K, MATH, GaoKao, CollegeMath and Olympiad datasets show that SCoT reduces reasoning latency by 48\%$\sim$66\% for Deepseek-R1-Distill-Qwen-32B while achieving near-target-model-level performance. Our code is available at this https URL. 

---
# WuNeng: Hybrid State with Attention 

**Authors**: Liu Xiao, Li Zhiyuan, Lin Yueyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19191)  

**Abstract**: The WuNeng architecture introduces a novel approach to enhancing the expressivity and power of large language models by integrating recurrent neural network (RNN)-based RWKV-7 with advanced attention mechanisms, prioritizing heightened contextual coherence over reducing KV cache size. Building upon the hybrid-head concept from Hymba, WuNeng augments standard multi-head attention with additional RWKV-7 state-driven heads, rather than replacing existing heads, to enrich the model's representational capacity. A cross-head interaction technique fosters dynamic synergy among standard, state-driven, and newly introduced middle heads, leveraging concatenation, additive modulation, and gated fusion for robust information integration. Furthermore, a multi-token state processing mechanism harnesses the continuous RWKV-7 state to capture intricate, sequence-wide dependencies, significantly boosting expressivity. Remarkably, these enhancements are achieved with minimal additional parameters, ensuring efficiency while empowering the model to excel in complex reasoning and sequence generation tasks. WuNeng sets a new standard for balancing expressivity and computational efficiency in modern neural architectures. 

---
# Effective Length Extrapolation via Dimension-Wise Positional Embeddings Manipulation 

**Authors**: Yi Lu, Wanxu Zhao, Xin Zhou, Chenxin An, Chenglong Wang, Shuo Li, Yuming Yang, Jun Zhao, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18857)  

**Abstract**: Large Language Models (LLMs) often struggle to process and generate coherent context when the number of input tokens exceeds the pre-trained length. Recent advancements in long-context extension have significantly expanded the context window of LLMs but require expensive overhead to train the large-scale models with longer context. In this work, we propose Dimension-Wise Positional Embeddings Manipulation (DPE), a training-free framework to extrapolate the context window of LLMs by diving into RoPE's different hidden dimensions. Instead of manipulating all dimensions equally, DPE detects the effective length for every dimension and finds the key dimensions for context extension. We reuse the original position indices with their embeddings from the pre-trained model and manipulate the key dimensions' position indices to their most effective lengths. In this way, DPE adjusts the pre-trained models with minimal modifications while ensuring that each dimension reaches its optimal state for extrapolation. DPE significantly surpasses well-known baselines such as YaRN and Self-Extend. DPE enables Llama3-8k 8B to support context windows of 128k tokens without continual training and integrates seamlessly with Flash Attention 2. In addition to its impressive extrapolation capability, DPE also dramatically improves the models' performance within training length, such as Llama3.1 70B, by over 18 points on popular long-context benchmarks RULER. When compared with commercial models, Llama 3.1 70B with DPE even achieves better performance than GPT-4-128K. 

---
# When2Call: When (not) to Call Tools 

**Authors**: Hayley Ross, Ameya Sunil Mahabaleshwarkar, Yoshi Suhara  

**Link**: [PDF](https://arxiv.org/pdf/2504.18851)  

**Abstract**: Leveraging external tools is a key feature for modern Language Models (LMs) to expand their capabilities and integrate them into existing systems. However, existing benchmarks primarily focus on the accuracy of tool calling -- whether the correct tool is called with the correct parameters -- and less on evaluating when LMs should (not) call tools. We develop a new benchmark, When2Call, which evaluates tool-calling decision-making: when to generate a tool call, when to ask follow-up questions and when to admit the question can't be answered with the tools provided. We find that state-of-the-art tool-calling LMs show significant room for improvement on When2Call, indicating the importance of this benchmark. We also develop a training set for When2Call and leverage the multiple-choice nature of the benchmark to develop a preference optimization training regime, which shows considerably more improvement than traditional fine-tuning. We release the benchmark and training data as well as evaluation scripts at this https URL. 

---
# Towards Robust Dialogue Breakdown Detection: Addressing Disruptors in Large Language Models with Self-Guided Reasoning 

**Authors**: Abdellah Ghassel, Xianzhi Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18839)  

**Abstract**: Large language models (LLMs) are rapidly changing various domains. However, their capabilities in handling conversational breakdowns still require an in-depth exploration. This paper addresses the challenge of detecting and mitigating dialogue breakdowns within LLM-driven conversational systems. While powerful models from OpenAI and Anthropic excel in many dialogue tasks, they can still produce incoherent or contradictory responses, commonly referred to as breakdowns, which undermine user trust. To tackle this, we propose an approach that combines specialized fine-tuning with advanced prompting strategies, including few-shot learning, chain-of-thought reasoning, and analogical prompting. In particular, we fine-tune a small 8B model and demonstrate its robust classification and calibration capabilities in English and Japanese dialogue. We also validate its generalization on the BETOLD dataset, achieving a 7\% accuracy improvement over its base model. Furthermore, we introduce a real-time deployment architecture that selectively escalates suspicious responses to more resource-intensive frontier models only when breakdowns are detected, significantly cutting operational expenses and energy consumption. Experimental results show our method surpasses prior state-of-the-art specialized classifiers while also narrowing performance gaps between smaller open-source models and large proprietary ones. Our approach offers a scalable solution for robust conversational AI in high-impact domains by combining efficiency, interpretability, and reliability. 

---
# Toward Generalizable Evaluation in the LLM Era: A Survey Beyond Benchmarks 

**Authors**: Yixin Cao, Shibo Hong, Xinze Li, Jiahao Ying, Yubo Ma, Haiyuan Liang, Yantao Liu, Zijun Yao, Xiaozhi Wang, Dan Huang, Wenxuan Zhang, Lifu Huang, Muhao Chen, Lei Hou, Qianru Sun, Xingjun Ma, Zuxuan Wu, Min-Yen Kan, David Lo, Qi Zhang, Heng Ji, Jing Jiang, Juanzi Li, Aixin Sun, Xuanjing Huang, Tat-Seng Chua, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18838)  

**Abstract**: Large Language Models (LLMs) are advancing at an amazing speed and have become indispensable across academia, industry, and daily applications. To keep pace with the status quo, this survey probes the core challenges that the rise of LLMs poses for evaluation. We identify and analyze two pivotal transitions: (i) from task-specific to capability-based evaluation, which reorganizes benchmarks around core competencies such as knowledge, reasoning, instruction following, multi-modal understanding, and safety; and (ii) from manual to automated evaluation, encompassing dynamic dataset curation and "LLM-as-a-judge" scoring.
Yet, even with these transitions, a crucial obstacle persists: the evaluation generalization issue. Bounded test sets cannot scale alongside models whose abilities grow seemingly without limit. We will dissect this issue, along with the core challenges of the above two transitions, from the perspectives of methods, datasets, evaluators, and metrics. Due to the fast evolving of this field, we will maintain a living GitHub repository (links are in each section) to crowd-source updates and corrections, and warmly invite contributors and collaborators. 

---
# KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation 

**Authors**: Jiabin Fan, Guoqing Luo, Michael Bowling, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2504.19024)  

**Abstract**: We propose a novel k-step return estimation method (called KETCHUP) for Reinforcement Learning(RL)-based knowledge distillation (KD) in text generation tasks. Our idea is to induce a K-step return by using the Bellman Optimality Equation for multiple steps. Theoretical analysis shows that this K-step formulation reduces the variance of the gradient estimates, thus leading to improved RL optimization especially when the student model size is large. Empirical evaluation on three text generation tasks demonstrates that our approach yields superior performance in both standard task metrics and large language model (LLM)-based evaluation. These results suggest that our K-step return induction offers a promising direction for enhancing RL-based KD in LLM research. 

---
# SynLexLM: Scaling Legal LLMs with Synthetic Data and Curriculum Learning 

**Authors**: Ojasw Upadhyay, Abishek Saravankumar, Ayman Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2504.18762)  

**Abstract**: Large Language Models (LLMs) are powerful but often require extensive fine-tuning and large datasets for specialized domains like law. General-purpose pre-training may not capture legal nuances, and acquiring sufficient legal data is challenging. We introduce SynLexLM, a novel approach to efficiently pre-train a legal LLM. Our method employs curriculum learning, progressing from simple to complex legal texts and queries, combined with synthetic data augmentation using models like Gemini Pro to address data scarcity. We aim to achieve improved performance on legal benchmarks (BigLaw-Bench, EUR-Lex-Sum) compared to traditional models and fine-tuned versions. Preliminary work involves generating synthetic QA pairs reflecting legal reasoning. This work aims to enhance legal document analysis and research tools, potentially democratizing access to advanced legal AI. 

---
# A Simple Ensemble Strategy for LLM Inference: Towards More Stable Text Classification 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18884)  

**Abstract**: With the advance of large language models (LLMs), LLMs have been utilized for the various tasks. However, the issues of variability and reproducibility of results from each trial of LLMs have been largely overlooked in existing literature while actual human annotation uses majority voting to resolve disagreements among annotators. Therefore, this study introduces the straightforward ensemble strategy to a sentiment analysis using LLMs. As the results, we demonstrate that the ensemble of multiple inference using medium-sized LLMs produces more robust and accurate results than using a large model with a single attempt with reducing RMSE by 18.6%. 

---
# Span-Level Hallucination Detection for LLM-Generated Answers 

**Authors**: Passant Elchafei, Mervet Abu-Elkheir  

**Link**: [PDF](https://arxiv.org/pdf/2504.18639)  

**Abstract**: Detecting spans of hallucination in LLM-generated answers is crucial for improving factual consistency. This paper presents a span-level hallucination detection framework for the SemEval-2025 Shared Task, focusing on English and Arabic texts. Our approach integrates Semantic Role Labeling (SRL) to decompose the answer into atomic roles, which are then compared with a retrieved reference context obtained via question-based LLM prompting. Using a DeBERTa-based textual entailment model, we evaluate each role semantic alignment with the retrieved context. The entailment scores are further refined through token-level confidence measures derived from output logits, and the combined scores are used to detect hallucinated spans. Experiments on the Mu-SHROOM dataset demonstrate competitive performance. Additionally, hallucinated spans have been verified through fact-checking by prompting GPT-4 and LLaMA. Our findings contribute to improving hallucination detection in LLM-generated responses. 

---
# Mind the Language Gap: Automated and Augmented Evaluation of Bias in LLMs for High- and Low-Resource Languages 

**Authors**: Alessio Buscemi, Cédric Lothritz, Sergio Morales, Marcos Gomez-Vazquez, Robert Clarisó, Jordi Cabot, German Castignani  

**Link**: [PDF](https://arxiv.org/pdf/2504.18560)  

**Abstract**: Large Language Models (LLMs) have exhibited impressive natural language processing capabilities but often perpetuate social biases inherent in their training data. To address this, we introduce MultiLingual Augmented Bias Testing (MLA-BiTe), a framework that improves prior bias evaluation methods by enabling systematic multilingual bias testing. MLA-BiTe leverages automated translation and paraphrasing techniques to support comprehensive assessments across diverse linguistic settings. In this study, we evaluate the effectiveness of MLA-BiTe by testing four state-of-the-art LLMs in six languages -- including two low-resource languages -- focusing on seven sensitive categories of discrimination. 

---
# Evaluate-and-Purify: Fortifying Code Language Models Against Adversarial Attacks Using LLM-as-a-Judge 

**Authors**: Wenhan Mu, Ling Xu, Shuren Pei, Le Mi, Huichi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.19730)  

**Abstract**: The widespread adoption of code language models in software engineering tasks has exposed vulnerabilities to adversarial attacks, especially the identifier substitution attacks. Although existing identifier substitution attackers demonstrate high success rates, they often produce adversarial examples with unnatural code patterns. In this paper, we systematically assess the quality of adversarial examples using LLM-as-a-Judge. Our analysis reveals that over 80% of adversarial examples generated by state-of-the-art identifier substitution attackers (e.g., ALERT) are actually detectable. Based on this insight, we propose EP-Shield, a unified framework for evaluating and purifying identifier substitution attacks via naturalness-aware reasoning. Specifically, we first evaluate the naturalness of code and identify the perturbed adversarial code, then purify it so that the victim model can restore correct prediction. Extensive experiments demonstrate the superiority of EP-Shield over adversarial fine-tuning (up to 83.36% improvement) and its lightweight design 7B parameters) with GPT-4-level performance. 

---
# LawFlow : Collecting and Simulating Lawyers' Thought Processes 

**Authors**: Debarati Das, Khanh Chi Le, Ritik Sachin Parkar, Karin De Langis, Brendan Madson, Chad M. Berryman, Robin M. Willis, Daniel H. Moses, Brett McDonnell, Daniel Schwarcz, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18942)  

**Abstract**: Legal practitioners, particularly those early in their careers, face complex, high-stakes tasks that require adaptive, context-sensitive reasoning. While AI holds promise in supporting legal work, current datasets and models are narrowly focused on isolated subtasks and fail to capture the end-to-end decision-making required in real-world practice. To address this gap, we introduce LawFlow, a dataset of complete end-to-end legal workflows collected from trained law students, grounded in real-world business entity formation scenarios. Unlike prior datasets focused on input-output pairs or linear chains of thought, LawFlow captures dynamic, modular, and iterative reasoning processes that reflect the ambiguity, revision, and client-adaptive strategies of legal practice. Using LawFlow, we compare human and LLM-generated workflows, revealing systematic differences in structure, reasoning flexibility, and plan execution. Human workflows tend to be modular and adaptive, while LLM workflows are more sequential, exhaustive, and less sensitive to downstream implications. Our findings also suggest that legal professionals prefer AI to carry out supportive roles, such as brainstorming, identifying blind spots, and surfacing alternatives, rather than executing complex workflows end-to-end. Building on these findings, we propose a set of design suggestions, rooted in empirical observations, that align AI assistance with human goals of clarity, completeness, creativity, and efficiency, through hybrid planning, adaptive execution, and decision-point support. Our results highlight both the current limitations of LLMs in supporting complex legal workflows and opportunities for developing more collaborative, reasoning-aware legal AI systems. All data and code are available on our project page (this https URL). 

---
# Improving Reasoning Performance in Large Language Models via Representation Engineering 

**Authors**: Bertram Højer, Oliver Jarvis, Stefan Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2504.19483)  

**Abstract**: Recent advancements in large language models (LLMs) have resulted in increasingly anthropomorphic language concerning the ability of LLMs to reason. Whether reasoning in LLMs should be understood to be inherently different is, however, widely debated. We propose utilizing a representation engineering approach wherein model activations are read from the residual stream of an LLM when processing a reasoning task. The activations are used to derive a control vector that is applied to the model as an inference-time intervention, modulating the representational space of the model, to improve performance on the specified task. We publish the code for deriving control vectors and analyzing model representations. The method allows us to improve performance on reasoning benchmarks and assess how control vectors influence the final logit distribution of a model via metrics such as KL divergence and entropy. We apply control vectors to Mistral-7B-Instruct and a range of Pythia models on an inductive, a deductive and mathematical reasoning task. We show that an LLM can, to a certain degree, be controlled to improve its perceived reasoning ability by modulating activations. The intervention is dependent upon the ability to reliably extract the model's typical state when correctly solving a task. Our results suggest that reasoning performance can be modulated in the same manner as other information-processing tasks performed by LLMs and demonstrate that we are capable of improving performance on specific tasks via a simple intervention on the residual stream with no additional training. 

---
# Large Language Models are Qualified Benchmark Builders: Rebuilding Pre-Training Datasets for Advancing Code Intelligence Tasks 

**Authors**: Kang Yang, Xinjun Mao, Shangwen Wang, Yanlin Wang, Tanghaoran Zhang, Bo Lin, Yihao Qin, Zhang Zhang, Yao Lu, Kamal Al-Sabahi  

**Link**: [PDF](https://arxiv.org/pdf/2504.19444)  

**Abstract**: Pre-trained code models rely heavily on high-quality pre-training data, particularly human-written reference comments that bridge code and natural language. However, these comments often become outdated as software evolves, degrading model performance. Large language models (LLMs) excel at generating high-quality code comments. We investigate whether replacing human-written comments with LLM-generated ones improves pre-training datasets. Since standard metrics cannot assess reference comment quality, we propose two novel reference-free evaluation tasks: code-comment inconsistency detection and semantic code search. Results show that LLM-generated comments are more semantically consistent with code than human-written ones, as confirmed by manual evaluation. Leveraging this finding, we rebuild the CodeSearchNet dataset with LLM-generated comments and re-pre-train CodeT5. Evaluations demonstrate that models trained on LLM-enhanced data outperform those using original human comments in code summarization, generation, and translation tasks. This work validates rebuilding pre-training datasets with LLMs to advance code intelligence, challenging the traditional reliance on human reference comments. 

---
# MTCSC: Retrieval-Augmented Iterative Refinement for Chinese Spelling Correction 

**Authors**: Junhong Liang, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18938)  

**Abstract**: Chinese Spelling Correction (CSC) aims to detect and correct erroneous tokens in sentences. While Large Language Models (LLMs) have shown remarkable success in identifying and rectifying potential errors, they often struggle with maintaining consistent output lengths and adapting to domain-specific corrections. Furthermore, existing CSC task impose rigid constraints requiring input and output lengths to be identical, limiting their applicability. In this work, we extend traditional CSC to variable-length correction scenarios, including Chinese Splitting Error Correction (CSEC) and ASR N-best Error Correction. To address domain adaptation and length consistency, we propose MTCSC (Multi-Turn CSC) framework based on RAG enhanced with a length reflection mechanism. Our approach constructs a retrieval database from domain-specific training data and dictionaries, fine-tuning retrievers to optimize performance for error-containing inputs. Additionally, we introduce a multi-source combination strategy with iterative length reflection to ensure output length fidelity. Experiments across diverse domain datasets demonstrate that our method significantly outperforms current approaches in correction quality, particularly in handling domain-specific and variable-length error correction tasks. 

---
# Stealing Creator's Workflow: A Creator-Inspired Agentic Framework with Iterative Feedback Loop for Improved Scientific Short-form Generation 

**Authors**: Jong Inn Park, Maanas Taneja, Qianwen Wang, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18805)  

**Abstract**: Generating engaging, accurate short-form videos from scientific papers is challenging due to content complexity and the gap between expert authors and readers. Existing end-to-end methods often suffer from factual inaccuracies and visual artifacts, limiting their utility for scientific dissemination. To address these issues, we propose SciTalk, a novel multi-LLM agentic framework, grounding videos in various sources, such as text, figures, visual styles, and avatars. Inspired by content creators' workflows, SciTalk uses specialized agents for content summarization, visual scene planning, and text and layout editing, and incorporates an iterative feedback mechanism where video agents simulate user roles to give feedback on generated videos from previous iterations and refine generation prompts. Experimental evaluations show that SciTalk outperforms simple prompting methods in generating scientifically accurate and engaging content over the refined loop of video generation. Although preliminary results are still not yet matching human creators' quality, our framework provides valuable insights into the challenges and benefits of feedback-driven video generation. Our code, data, and generated videos will be publicly available. 

---
# ClimaEmpact: Domain-Aligned Small Language Models and Datasets for Extreme Weather Analytics 

**Authors**: Deeksha Varshney, Keane Ong, Rui Mao, Erik Cambria, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2504.19066)  

**Abstract**: Accurate assessments of extreme weather events are vital for research and policy, yet localized and granular data remain scarce in many parts of the world. This data gap limits our ability to analyze potential outcomes and implications of extreme weather events, hindering effective decision-making. Large Language Models (LLMs) can process vast amounts of unstructured text data, extract meaningful insights, and generate detailed assessments by synthesizing information from multiple sources. Furthermore, LLMs can seamlessly transfer their general language understanding to smaller models, enabling these models to retain key knowledge while being fine-tuned for specific tasks. In this paper, we propose Extreme Weather Reasoning-Aware Alignment (EWRA), a method that enhances small language models (SLMs) by incorporating structured reasoning paths derived from LLMs, and ExtremeWeatherNews, a large dataset of extreme weather event-related news articles. EWRA and ExtremeWeatherNews together form the overall framework, ClimaEmpact, that focuses on addressing three critical extreme-weather tasks: categorization of tangible vulnerabilities/impacts, topic labeling, and emotion analysis. By aligning SLMs with advanced reasoning strategies on ExtremeWeatherNews (and its derived dataset ExtremeAlign used specifically for SLM alignment), EWRA improves the SLMs' ability to generate well-grounded and domain-specific responses for extreme weather analytics. Our results show that the approach proposed guides SLMs to output domain-aligned responses, surpassing the performance of task-specific models and offering enhanced real-world applicability for extreme weather analytics. 

---
# Can Third-parties Read Our Emotions? 

**Authors**: Jiayi Li, Yingfan Zhou, Pranav Narayanan Venkit, Halima Binte Islam, Sneha Arya, Shomir Wilson, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18673)  

**Abstract**: Natural Language Processing tasks that aim to infer an author's private states, e.g., emotions and opinions, from their written text, typically rely on datasets annotated by third-party annotators. However, the assumption that third-party annotators can accurately capture authors' private states remains largely unexamined. In this study, we present human subjects experiments on emotion recognition tasks that directly compare third-party annotations with first-party (author-provided) emotion labels. Our findings reveal significant limitations in third-party annotations-whether provided by human annotators or large language models (LLMs)-in faithfully representing authors' private states. However, LLMs outperform human annotators nearly across the board. We further explore methods to improve third-party annotation quality. We find that demographic similarity between first-party authors and third-party human annotators enhances annotation performance. While incorporating first-party demographic information into prompts leads to a marginal but statistically significant improvement in LLMs' performance. We introduce a framework for evaluating the limitations of third-party annotations and call for refined annotation practices to accurately represent and model authors' private states. 

---
# Generative Product Recommendations for Implicit Superlative Queries 

**Authors**: Kaustubh D. Dhole, Nikhita Vedula, Saar Kuzi, Giuseppe Castellucci, Eugene Agichtein, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18748)  

**Abstract**: In Recommender Systems, users often seek the best products through indirect, vague, or under-specified queries, such as "best shoes for trail running". Such queries, also referred to as implicit superlative queries, pose a significant challenge for standard retrieval and ranking systems as they lack an explicit mention of attributes and require identifying and reasoning over complex factors. We investigate how Large Language Models (LLMs) can generate implicit attributes for ranking as well as reason over them to improve product recommendations for such queries. As a first step, we propose a novel four-point schema for annotating the best product candidates for superlative queries called SUPERB, paired with LLM-based product annotations. We then empirically evaluate several existing retrieval and ranking approaches on our new dataset, providing insights and discussing their integration into real-world e-commerce production systems. 

---
# Hierarchical Attention Generates Better Proofs 

**Authors**: Jianlong Chen, Chao Li, Yang Yuan, Andrew C Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.19188)  

**Abstract**: Large language models (LLMs) have shown promise in formal theorem proving, but their token-level processing often fails to capture the inherent hierarchical nature of mathematical proofs. We introduce \textbf{Hierarchical Attention}, a regularization method that aligns LLMs' attention mechanisms with mathematical reasoning structures. Our approach establishes a five-level hierarchy from foundational elements to high-level concepts, ensuring structured information flow in proof generation. Experiments demonstrate that our method improves proof success rates by 2.05\% on miniF2F and 1.69\% on ProofNet while reducing proof complexity by 23.81\% and 16.50\% respectively. The code is available at this https URL. 

---
# AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space of Language Embeddings 

**Authors**: Guoqing Hu, An Zhang, Shuo Liu, Zhibo Cai, Xun Yang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19218)  

**Abstract**: Recent advancements in sequential recommendation have underscored the potential of Large Language Models (LLMs) for enhancing item embeddings. However, existing approaches face three key limitations: 1) the degradation of the semantic space when high-dimensional language embeddings are mapped to lower-dimensional ID embeddings, 2) the underutilization of language embeddings, and 3) the reliance on additional trainable parameters, such as an adapter, to bridge the gap between the semantic and behavior this http URL this paper, we introduce AlphaFuse, a simple but effective language-guided learning strategy that addresses these challenges by learning ID embeddings within the null space of language embeddings. Specifically, we decompose the semantic space of language embeddings via Singular Value Decomposition (SVD), distinguishing it into a semantic-rich row space and a semantic-sparse null space. Collaborative signals are then injected into the null space, while preserving the rich semantics of the row space. AlphaFuse prevents degradation of the semantic space, integrates the retained language embeddings into the final item embeddings, and eliminates the need for auxiliary trainable modules, enabling seamless adaptation to any sequential recommendation framework. We validate the effectiveness and flexibility of AlphaFuse through extensive experiments on three benchmark datasets, including cold-start user and long-tail settings, showcasing significant improvements in both discriminative and diffusion-based generative sequential recommenders. Our codes and datasets are available at this https URL. 

---
# LLM-Evaluation Tropes: Perspectives on the Validity of LLM-Evaluations 

**Authors**: Laura Dietz, Oleg Zendel, Peter Bailey, Charles Clarke, Ellese Cotterill, Jeff Dalton, Faegheh Hasibi, Mark Sanderson, Nick Craswell  

**Link**: [PDF](https://arxiv.org/pdf/2504.19076)  

**Abstract**: Large Language Models (LLMs) are increasingly used to evaluate information retrieval (IR) systems, generating relevance judgments traditionally made by human assessors. Recent empirical studies suggest that LLM-based evaluations often align with human judgments, leading some to suggest that human judges may no longer be necessary, while others highlight concerns about judgment reliability, validity, and long-term impact. As IR systems begin incorporating LLM-generated signals, evaluation outcomes risk becoming self-reinforcing, potentially leading to misleading conclusions.
This paper examines scenarios where LLM-evaluators may falsely indicate success, particularly when LLM-based judgments influence both system development and evaluation. We highlight key risks, including bias reinforcement, reproducibility challenges, and inconsistencies in assessment methodologies. To address these concerns, we propose tests to quantify adverse effects, guardrails, and a collaborative framework for constructing reusable test collections that integrate LLM judgments responsibly. By providing perspectives from academia and industry, this work aims to establish best practices for the principled use of LLMs in IR evaluation. 

---
# Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses 

**Authors**: Sahel Sharifymoghaddam, Shivani Upadhyay, Nandan Thakur, Ronak Pradeep, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.20006)  

**Abstract**: Battles, or side-by-side comparisons in so called arenas that elicit human preferences, have emerged as a popular approach to assessing the output quality of LLMs. Recently, this idea has been extended to retrieval-augmented generation (RAG) systems. While undoubtedly representing an advance in evaluation, battles have at least two drawbacks, particularly in the context of complex information-seeking queries: they are neither explanatory nor diagnostic. Recently, the nugget evaluation methodology has emerged as a promising approach to evaluate the quality of RAG answers. Nuggets decompose long-form LLM-generated answers into atomic facts, highlighting important pieces of information necessary in a "good" response. In this work, we apply our AutoNuggetizer framework to analyze data from roughly 7K Search Arena battles provided by LMArena in a fully automatic manner. Our results show a significant correlation between nugget scores and human preferences, showcasing promise in our approach to explainable and diagnostic system evaluations. 

---
# From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review 

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2504.19678)  

**Abstract**: Large language models and autonomous AI agents have evolved rapidly, resulting in a diverse array of evaluation benchmarks, frameworks, and collaboration protocols. However, the landscape remains fragmented and lacks a unified taxonomy or comprehensive survey. Therefore, we present a side-by-side comparison of benchmarks developed between 2019 and 2025 that evaluate these models and agents across multiple domains. In addition, we propose a taxonomy of approximately 60 benchmarks that cover general and academic knowledge reasoning, mathematical problem-solving, code generation and software engineering, factual grounding and retrieval, domain-specific evaluations, multimodal and embodied tasks, task orchestration, and interactive assessments. Furthermore, we review AI-agent frameworks introduced between 2023 and 2025 that integrate large language models with modular toolkits to enable autonomous decision-making and multi-step reasoning. Moreover, we present real-world applications of autonomous AI agents in materials science, biomedical research, academic ideation, software engineering, synthetic data generation, chemical reasoning, mathematical problem-solving, geographic information systems, multimedia, healthcare, and finance. We then survey key agent-to-agent collaboration protocols, namely the Agent Communication Protocol (ACP), the Model Context Protocol (MCP), and the Agent-to-Agent Protocol (A2A). Finally, we discuss recommendations for future research, focusing on advanced reasoning strategies, failure modes in multi-agent LLM systems, automated scientific discovery, dynamic tool integration via reinforcement learning, integrated search capabilities, and security vulnerabilities in agent protocols. 

---
# Towards Automated Scoping of AI for Social Good Projects 

**Authors**: Jacob Emmerson, Rayid Ghani, Zheyuan Ryan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20010)  

**Abstract**: Artificial Intelligence for Social Good (AI4SG) is an emerging effort that aims to address complex societal challenges with the powerful capabilities of AI systems. These challenges range from local issues with transit networks to global wildlife preservation. However, regardless of scale, a critical bottleneck for many AI4SG initiatives is the laborious process of problem scoping -- a complex and resource-intensive task -- due to a scarcity of professionals with both technical and domain expertise. Given the remarkable applications of large language models (LLM), we propose a Problem Scoping Agent (PSA) that uses an LLM to generate comprehensive project proposals grounded in scientific literature and real-world knowledge. We demonstrate that our PSA framework generates proposals comparable to those written by experts through a blind review and AI evaluations. Finally, we document the challenges of real-world problem scoping and note several areas for future work. 

---
# The Convergent Ethics of AI? Analyzing Moral Foundation Priorities in Large Language Models with a Multi-Framework Approach 

**Authors**: Chad Coleman, W. Russell Neuman, Ali Dasdan, Safinah Ali, Manan Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.19255)  

**Abstract**: As large language models (LLMs) are increasingly deployed in consequential decision-making contexts, systematically assessing their ethical reasoning capabilities becomes a critical imperative. This paper introduces the Priorities in Reasoning and Intrinsic Moral Evaluation (PRIME) framework--a comprehensive methodology for analyzing moral priorities across foundational ethical dimensions including consequentialist-deontological reasoning, moral foundations theory, and Kohlberg's developmental stages. We apply this framework to six leading LLMs through a dual-protocol approach combining direct questioning and response analysis to established ethical dilemmas. Our analysis reveals striking patterns of convergence: all evaluated models demonstrate strong prioritization of care/harm and fairness/cheating foundations while consistently underweighting authority, loyalty, and sanctity dimensions. Through detailed examination of confidence metrics, response reluctance patterns, and reasoning consistency, we establish that contemporary LLMs (1) produce decisive ethical judgments, (2) demonstrate notable cross-model alignment in moral decision-making, and (3) generally correspond with empirically established human moral preferences. This research contributes a scalable, extensible methodology for ethical benchmarking while highlighting both the promising capabilities and systematic limitations in current AI moral reasoning architectures--insights critical for responsible development as these systems assume increasingly significant societal roles. 

---
# Fitness Landscape of Large Language Model-Assisted Automated Algorithm Search 

**Authors**: Fei Liu, Qingfu Zhang, Xialiang Tong, Mingxuan Yuan, Kun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.19636)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in algorithm design. However, when integrated into search frameworks for iterative algorithm search, the underlying fitness landscape--critical for understanding search behaviou--remains underexplored. In this paper, we illustrate and analyze the fitness landscape of LLM-assisted Algorithm Search (LAS) using a graph-based approach, where nodes represent algorithms and edges denote transitions between them. We conduct extensive evaluations across six algorithm design tasks and six commonly used LLMs. Our findings reveal that LAS landscapes are highly multimodal and rugged, particularly in combinatorial optimization tasks, with distinct structural variations across tasks and LLMs. For instance, heuristic design tasks exhibit dense clusters of high-performing algorithms, while symbolic regression tasks show sparse, scattered distributions. Additionally, we demonstrate how population size influences exploration-exploitation trade-offs and the evolving trajectory of elite algorithms. These insights not only advance our understanding of LAS landscapes but also provide practical guidance for designing more effective LAS methods. 

---
# GLaMoR: Consistency Checking of OWL Ontologies using Graph Language Models 

**Authors**: Justin Mücke, Ansgar Scherp  

**Link**: [PDF](https://arxiv.org/pdf/2504.19023)  

**Abstract**: Semantic reasoning aims to infer new knowledge from existing knowledge, with OWL ontologies serving as a standardized framework for organizing information. A key challenge in semantic reasoning is verifying ontology consistency. However, state-of-the-art reasoners are computationally expensive, and their efficiency decreases as ontology sizes grow. While classical machine learning models have been explored for consistency checking, they struggle to capture complex relationships within ontologies. Large language models (LLMs) have shown promising results for simple reasoning tasks but perform poorly on structured reasoning. The recently introduced Graph Language Model (GLM) offers a way to simultaneously process graph-structured data and text. This paper proposes GLaMoR (Graph Language Model for Reasoning), a reasoning pipeline that transforms OWL ontologies into graph-structured data and adapts the GLM architecture for consistency checking. We evaluate GLaMoR on ontologies from the NCBO BioPortal repository, converting them into triples suitable for model input. Our results show that the GLM outperforms all baseline models, achieving $95\%$ accuracy while being 20 times faster than classical reasoners.
The Code is accessible under: this https URL 

---
# Can AI Agents Design and Implement Drug Discovery Pipelines? 

**Authors**: Khachik Smbatyan, Tsolak Ghukasyan, Tigran Aghajanyan, Hovhannes Dabaghyan, Sergey Adamyan, Aram Bughdaryan, Vahagn Altunyan, Gagik Navasardyan, Aram Davtyan, Anush Hakobyan, Aram Gharibyan, Arman Fahradyan, Artur Hakobyan, Hasmik Mnatsakanyan, Narek Ginoyan, Garik Petrosyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.19912)  

**Abstract**: The rapid advancement of artificial intelligence, particularly autonomous agentic systems based on Large Language Models (LLMs), presents new opportunities to accelerate drug discovery by improving in-silico modeling and reducing dependence on costly experimental trials. Current AI agent-based systems demonstrate proficiency in solving programming challenges and conducting research, indicating an emerging potential to develop software capable of addressing complex problems such as pharmaceutical design and drug discovery. This paper introduces DO Challenge, a benchmark designed to evaluate the decision-making abilities of AI agents in a single, complex problem resembling virtual screening scenarios. The benchmark challenges systems to independently develop, implement, and execute efficient strategies for identifying promising molecular structures from extensive datasets, while navigating chemical space, selecting models, and managing limited resources in a multi-objective context. We also discuss insights from the DO Challenge 2025, a competition based on the proposed benchmark, which showcased diverse strategies explored by human participants. Furthermore, we present the Deep Thought multi-agent system, which demonstrated strong performance on the benchmark, outperforming most human teams. Among the language models tested, Claude 3.7 Sonnet, Gemini 2.5 Pro and o3 performed best in primary agent roles, and GPT-4o, Gemini 2.0 Flash were effective in auxiliary roles. While promising, the system's performance still fell short of expert-designed solutions and showed high instability, highlighting both the potential and current limitations of AI-driven methodologies in transforming drug discovery and broader scientific research. 

---
# Exploring a Large Language Model for Transforming Taxonomic Data into OWL: Lessons Learned and Implications for Ontology Development 

**Authors**: Filipi Miranda Soares, Antonio Mauro Saraiva, Luís Ferreira Pires, Luiz Olavo Bonino da Silva Santos, Dilvan de Abreu Moreira, Fernando Elias Corrêa, Kelly Rosa Braghetto, Debora Pignatari Drucker, Alexandre Cláudio Botazzo Delbem  

**Link**: [PDF](https://arxiv.org/pdf/2504.18651)  

**Abstract**: Managing scientific names in ontologies that represent species taxonomies is challenging due to the ever-evolving nature of these taxonomies. Manually maintaining these names becomes increasingly difficult when dealing with thousands of scientific names. To address this issue, this paper investigates the use of ChatGPT-4 to automate the development of the :Organism module in the Agricultural Product Types Ontology (APTO) for species classification. Our methodology involved leveraging ChatGPT-4 to extract data from the GBIF Backbone API and generate OWL files for further integration in APTO. Two alternative approaches were explored: (1) issuing a series of prompts for ChatGPT-4 to execute tasks via the BrowserOP plugin and (2) directing ChatGPT-4 to design a Python algorithm to perform analogous tasks. Both approaches rely on a prompting method where we provide instructions, context, input data, and an output indicator. The first approach showed scalability limitations, while the second approach used the Python algorithm to overcome these challenges, but it struggled with typographical errors in data handling. This study highlights the potential of Large language models like ChatGPT-4 to streamline the management of species names in ontologies. Despite certain limitations, these tools offer promising advancements in automating taxonomy-related tasks and improving the efficiency of ontology development. 

---
# ChiseLLM: Unleashing the Power of Reasoning LLMs for Chisel Agile Hardware Development 

**Authors**: Bowei Wang, Jiaran Gao, Yelai Feng, Renzhi Chen, Shanshan Li, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19144)  

**Abstract**: The growing demand for Domain-Specific Architecture (DSA) has driven the development of Agile Hardware Development Methodology (AHDM). Hardware Construction Language (HCL) like Chisel offers high-level abstraction features, making it an ideal language for HCL-Based AHDM. While Large Language Models (LLMs) excel in code generation tasks, they still face challenges with Chisel generation, particularly regarding syntax correctness and design variability. Recent reasoning models have significantly enhanced code generation capabilities through test-time scaling techniques. However, we found that reasoning models without domain adaptation cannot bring substantial benefits to Chisel code generation tasks. This paper presents ChiseLLM, a solution comprising data processing and transformation, prompt-guided reasoning trace synthesis, and domain-adapted model training. We constructed high-quality datasets from public RTL code resources and guided the model to adopt structured thinking patterns through prompt enhancement methods. Experiments demonstrate that our ChiseLLM-7B and ChiseLLM-32B models improved syntax correctness by 18.85% and 26.32% respectively over base models, while increasing variability design ability by 47.58% compared to baseline reasoning models. Our datasets and models are publicly available, providing high-performance, cost-effective models for HCL-Based AHDM, and offering an effective baseline for future research. Github repository: this https URL 

---
# BELL: Benchmarking the Explainability of Large Language Models 

**Authors**: Syed Quiser Ahmed, Bharathi Vokkaliga Ganesh, Jagadish Babu P, Karthick Selvaraj, ReddySiva Naga Parvathi Devi, Sravya Kappala  

**Link**: [PDF](https://arxiv.org/pdf/2504.18572)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in natural language processing, yet their decision-making processes often lack transparency. This opaqueness raises significant concerns regarding trust, bias, and model performance. To address these issues, understanding and evaluating the interpretability of LLMs is crucial. This paper introduces a standardised benchmarking technique, Benchmarking the Explainability of Large Language Models, designed to evaluate the explainability of large language models. 

---
# Modular Machine Learning: An Indispensable Path towards New-Generation Large Language Models 

**Authors**: Xin Wang, Haoyang Li, Zeyang Zhang, Haibo Chen, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20020)  

**Abstract**: Large language models (LLMs) have dramatically advanced machine learning research including natural language processing, computer vision, data mining, etc., yet they still exhibit critical limitations in reasoning, factual consistency, and interpretability. In this paper, we introduce a novel learning paradigm -- Modular Machine Learning (MML) -- as an essential approach toward new-generation LLMs. MML decomposes the complex structure of LLMs into three interdependent components: modular representation, modular model, and modular reasoning, aiming to enhance LLMs' capability of counterfactual reasoning, mitigating hallucinations, as well as promoting fairness, safety, and transparency. Specifically, the proposed MML paradigm can: i) clarify the internal working mechanism of LLMs through the disentanglement of semantic components; ii) allow for flexible and task-adaptive model design; iii) enable interpretable and logic-driven decision-making process. We present a feasible implementation of MML-based LLMs via leveraging advanced techniques such as disentangled representation learning, neural architecture search and neuro-symbolic learning. We critically identify key challenges, such as the integration of continuous neural and discrete symbolic processes, joint optimization, and computational scalability, present promising future research directions that deserve further exploration. Ultimately, the integration of the MML paradigm with LLMs has the potential to bridge the gap between statistical (deep) learning and formal (logical) reasoning, thereby paving the way for robust, adaptable, and trustworthy AI systems across a wide range of real-world applications. 

---
# Proof-of-TBI -- Fine-Tuned Vision Language Model Consortium and OpenAI-o3 Reasoning LLM-Based Medical Diagnosis Support System for Mild Traumatic Brain Injury (TBI) Prediction 

**Authors**: Ross Gore, Eranga Bandara, Sachin Shetty, Alberto E. Musto, Pratip Rana, Ambrosio Valencia-Romero, Christopher Rhea, Lobat Tayebi, Heather Richter, Atmaram Yarlagadda, Donna Edmonds, Steven Wallace, Donna Broshek  

**Link**: [PDF](https://arxiv.org/pdf/2504.18671)  

**Abstract**: Mild Traumatic Brain Injury (TBI) detection presents significant challenges due to the subtle and often ambiguous presentation of symptoms in medical imaging, making accurate diagnosis a complex task. To address these challenges, we propose Proof-of-TBI, a medical diagnosis support system that integrates multiple fine-tuned vision-language models with the OpenAI-o3 reasoning large language model (LLM). Our approach fine-tunes multiple vision-language models using a labeled dataset of TBI MRI scans, training them to diagnose TBI symptoms effectively. The predictions from these models are aggregated through a consensus-based decision-making process. The system evaluates the predictions from all fine-tuned vision language models using the OpenAI-o3 reasoning LLM, a model that has demonstrated remarkable reasoning performance, to produce the most accurate final diagnosis. The LLM Agents orchestrates interactions between the vision-language models and the reasoning LLM, managing the final decision-making process with transparency, reliability, and automation. This end-to-end decision-making workflow combines the vision-language model consortium with the OpenAI-o3 reasoning LLM, enabled by custom prompt engineering by the LLM agents. The prototype for the proposed platform was developed in collaboration with the U.S. Army Medical Research team in Newport News, Virginia, incorporating five fine-tuned vision-language models. The results demonstrate the transformative potential of combining fine-tuned vision-language model inputs with the OpenAI-o3 reasoning LLM to create a robust, secure, and highly accurate diagnostic system for mild TBI prediction. To the best of our knowledge, this research represents the first application of fine-tuned vision-language models integrated with a reasoning LLM for TBI prediction tasks. 

---
# Enhancing Surgical Documentation through Multimodal Visual-Temporal Transformers and Generative AI 

**Authors**: Hugo Georgenthum, Cristian Cosentino, Fabrizio Marozzo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.19918)  

**Abstract**: The automatic summarization of surgical videos is essential for enhancing procedural documentation, supporting surgical training, and facilitating post-operative analysis. This paper presents a novel method at the intersection of artificial intelligence and medicine, aiming to develop machine learning models with direct real-world applications in surgical contexts. We propose a multi-modal framework that leverages recent advancements in computer vision and large language models to generate comprehensive video summaries. %
The approach is structured in three key stages. First, surgical videos are divided into clips, and visual features are extracted at the frame level using visual transformers. This step focuses on detecting tools, tissues, organs, and surgical actions. Second, the extracted features are transformed into frame-level captions via large language models. These are then combined with temporal features, captured using a ViViT-based encoder, to produce clip-level summaries that reflect the broader context of each video segment. Finally, the clip-level descriptions are aggregated into a full surgical report using a dedicated LLM tailored for the summarization task. %
We evaluate our method on the CholecT50 dataset, using instrument and action annotations from 50 laparoscopic videos. The results show strong performance, achieving 96\% precision in tool detection and a BERT score of 0.74 for temporal context summarization. This work contributes to the advancement of AI-assisted tools for surgical reporting, offering a step toward more intelligent and reliable clinical documentation. 

---
# PhenoAssistant: A Conversational Multi-Agent AI System for Automated Plant Phenotyping 

**Authors**: Feng Chen, Ilias Stogiannidis, Andrew Wood, Danilo Bueno, Dominic Williams, Fraser Macfarlane, Bruce Grieve, Darren Wells, Jonathan A. Atkinson, Malcolm J. Hawkesford, Stephen A. Rolfe, Tracy Lawson, Tony Pridmore, Mario Valerio Giuffrida, Sotirios A. Tsaftaris  

**Link**: [PDF](https://arxiv.org/pdf/2504.19818)  

**Abstract**: Plant phenotyping increasingly relies on (semi-)automated image-based analysis workflows to improve its accuracy and scalability. However, many existing solutions remain overly complex, difficult to reimplement and maintain, and pose high barriers for users without substantial computational expertise. To address these challenges, we introduce PhenoAssistant: a pioneering AI-driven system that streamlines plant phenotyping via intuitive natural language interaction. PhenoAssistant leverages a large language model to orchestrate a curated toolkit supporting tasks including automated phenotype extraction, data visualisation and automated model training. We validate PhenoAssistant through several representative case studies and a set of evaluation tasks. By significantly lowering technical hurdles, PhenoAssistant underscores the promise of AI-driven methodologies to democratising AI adoption in plant biology. 

---
# Reshaping MOFs Text Mining with a Dynamic Multi-Agent Framework of Large Language Agents 

**Authors**: Zuhong Lin, Daoyuan Ren, Kai Ran, Sun Jing, Xiaotiang Huang, Haiyang He, Pengxu Pan, Xiaohang Zhang, Ying Fang, Tianying Wang, Minli Wu, Zhanglin Li, Xiaochuan Zhang, Haipu Li, Jingjing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.18880)  

**Abstract**: The mining of synthesis conditions for metal-organic frameworks (MOFs) is a significant focus in materials science. However, identifying the precise synthesis conditions for specific MOFs within the vast array of possibilities presents a considerable challenge. Large Language Models (LLMs) offer a promising solution to this problem. We leveraged the capabilities of LLMs, specifically gpt-4o-mini, as core agents to integrate various MOF-related agents, including synthesis, attribute, and chemical information agents. This integration culminated in the development of MOFh6, an LLM tool designed to streamline the MOF synthesis process. MOFh6 allows users to query in multiple formats, such as submitting scientific literature, or inquiring about specific MOF codes or structural properties. The tool analyzes these queries to provide optimal synthesis conditions and generates model files for density functional theory pre modeling. We believe MOFh6 will enhance efficiency in the MOF synthesis of all researchers. 

---
# TD-EVAL: Revisiting Task-Oriented Dialogue Evaluation by Combining Turn-Level Precision with Dialogue-Level Comparisons 

**Authors**: Emre Can Acikgoz, Carl Guo, Suvodip Dey, Akul Datta, Takyoung Kim, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2504.19982)  

**Abstract**: Task-oriented dialogue (TOD) systems are experiencing a revolution driven by Large Language Models (LLMs), yet the evaluation methodologies for these systems remain insufficient for their growing sophistication. While traditional automatic metrics effectively assessed earlier modular systems, they focus solely on the dialogue level and cannot detect critical intermediate errors that can arise during user-agent interactions. In this paper, we introduce TD-EVAL (Turn and Dialogue-level Evaluation), a two-step evaluation framework that unifies fine-grained turn-level analysis with holistic dialogue-level comparisons. At turn level, we evaluate each response along three TOD-specific dimensions: conversation cohesion, backend knowledge consistency, and policy compliance. Meanwhile, we design TOD Agent Arena that uses pairwise comparisons to provide a measure of dialogue-level quality. Through experiments on MultiWOZ 2.4 and {\tau}-Bench, we demonstrate that TD-EVAL effectively identifies the conversational errors that conventional metrics miss. Furthermore, TD-EVAL exhibits better alignment with human judgments than traditional and LLM-based metrics. These findings demonstrate that TD-EVAL introduces a new paradigm for TOD system evaluation, efficiently assessing both turn and system levels with a plug-and-play framework for future research. 

---
# $\texttt{SAGE}$: A Generic Framework for LLM Safety Evaluation 

**Authors**: Madhur Jindal, Hari Shrawgi, Parag Agrawal, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2504.19674)  

**Abstract**: Safety evaluation of Large Language Models (LLMs) has made progress and attracted academic interest, but it remains challenging to keep pace with the rapid integration of LLMs across diverse applications. Different applications expose users to various harms, necessitating application-specific safety evaluations with tailored harms and policies. Another major gap is the lack of focus on the dynamic and conversational nature of LLM systems. Such potential oversights can lead to harms that go unnoticed in standard safety benchmarks. This paper identifies the above as key requirements for robust LLM safety evaluation and recognizing that current evaluation methodologies do not satisfy these, we introduce the $\texttt{SAGE}$ (Safety AI Generic Evaluation) framework. $\texttt{SAGE}$ is an automated modular framework designed for customized and dynamic harm evaluations. It utilizes adversarial user models that are system-aware and have unique personalities, enabling a holistic red-teaming evaluation. We demonstrate $\texttt{SAGE}$'s effectiveness by evaluating seven state-of-the-art LLMs across three applications and harm policies. Our experiments with multi-turn conversational evaluations revealed a concerning finding that harm steadily increases with conversation length. Furthermore, we observe significant disparities in model behavior when exposed to different user personalities and scenarios. Our findings also reveal that some models minimize harmful outputs by employing severe refusal tactics that can hinder their usefulness. These insights highlight the necessity of adaptive and context-specific testing to ensure better safety alignment and safer deployment of LLMs in real-world scenarios. 

---
# A Tripartite Perspective on GraphRAG 

**Authors**: Michael Banf, Johannes Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2504.19667)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various domains, yet they struggle with knowledge-intensive tasks in areas that demand factual accuracy, e.g. industrial automation and healthcare. Key limitations include their tendency to hallucinate, lack of source traceability (provenance), and challenges in timely knowledge updates. Combining language models with knowledge graphs (GraphRAG) offers promising avenues for overcoming these deficits. However, a major challenge lies in creating such a knowledge graph in the first place. Here, we propose a novel approach that combines LLMs with a tripartite knowledge graph representation, which is constructed by connecting complex, domain-specific objects via a curated ontology of corresponding, domain-specific concepts to relevant sections within chunks of text through a concept-anchored pre-analysis of source documents starting from an initial lexical graph. As a consequence, our Tripartite-GraphRAG approach implements: i) a concept-specific, information-preserving pre-compression of textual chunks; ii) allows for the formation of a concept-specific relevance estimation of embedding similarities grounded in statistics; and iii) avoids common challenges w.r.t. continuous extendability, such as the need for entity resolution and deduplication. By applying a transformation to the knowledge graph, we formulate LLM prompt creation as an unsupervised node classification problem, drawing on ideas from Markov Random Fields. We evaluate our approach on a healthcare use case, involving multi-faceted analyses of patient anamneses given a set of medical concepts as well as clinical literature. Experiments indicate that it can optimize information density, coverage, and arrangement of LLM prompts while reducing their lengths, which may lead to reduced costs and more consistent and reliable LLM outputs. 

---
# An Automated Reinforcement Learning Reward Design Framework with Large Language Model for Cooperative Platoon Coordination 

**Authors**: Dixiao Wei, Peng Yi, Jinlong Lei, Yiguang Hong, Yuchuan Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.19480)  

**Abstract**: Reinforcement Learning (RL) has demonstrated excellent decision-making potential in platoon coordination problems. However, due to the variability of coordination goals, the complexity of the decision problem, and the time-consumption of trial-and-error in manual design, finding a well performance reward function to guide RL training to solve complex platoon coordination problems remains challenging. In this paper, we formally define the Platoon Coordination Reward Design Problem (PCRDP), extending the RL-based cooperative platoon coordination problem to incorporate automated reward function generation. To address PCRDP, we propose a Large Language Model (LLM)-based Platoon coordination Reward Design (PCRD) framework, which systematically automates reward function discovery through LLM-driven initialization and iterative optimization. In this method, LLM first initializes reward functions based on environment code and task requirements with an Analysis and Initial Reward (AIR) module, and then iteratively optimizes them based on training feedback with an evolutionary module. The AIR module guides LLM to deepen their understanding of code and tasks through a chain of thought, effectively mitigating hallucination risks in code generation. The evolutionary module fine-tunes and reconstructs the reward function, achieving a balance between exploration diversity and convergence stability for training. To validate our approach, we establish six challenging coordination scenarios with varying complexity levels within the Yangtze River Delta transportation network simulation. Comparative experimental results demonstrate that RL agents utilizing PCRD-generated reward functions consistently outperform human-engineered reward functions, achieving an average of 10\% higher performance metrics in all scenarios. 

---
# LLMs for Engineering: Teaching Models to Design High Powered Rockets 

**Authors**: Toby Simonds  

**Link**: [PDF](https://arxiv.org/pdf/2504.19394)  

**Abstract**: Large Language Models (LLMs) have transformed software engineering, but their application to physical engineering domains remains underexplored. This paper evaluates LLMs' capabilities in high-powered rocketry design through RocketBench, a benchmark connecting LLMs to high-fidelity rocket simulations. We test models on two increasingly complex design tasks: target altitude optimization and precision landing challenges. Our findings reveal that while state-of-the-art LLMs demonstrate strong baseline engineering knowledge, they struggle to iterate on their designs when given simulation results and ultimately plateau below human performance levels. However, when enhanced with reinforcement learning (RL), we show that a 7B parameter model outperforms both SoTA foundation models and human experts. This research demonstrates that RL-trained LLMs can serve as effective tools for complex engineering optimization, potentially transforming engineering domains beyond software development. 

---
# From Inductive to Deductive: LLMs-Based Qualitative Data Analysis in Requirements Engineering 

**Authors**: Syed Tauhid Ullah Shah, Mohamad Hussein, Ann Barcomb, Mohammad Moshirpour  

**Link**: [PDF](https://arxiv.org/pdf/2504.19384)  

**Abstract**: Requirements Engineering (RE) is essential for developing complex and regulated software projects. Given the challenges in transforming stakeholder inputs into consistent software designs, Qualitative Data Analysis (QDA) provides a systematic approach to handling free-form data. However, traditional QDA methods are time-consuming and heavily reliant on manual effort. In this paper, we explore the use of Large Language Models (LLMs), including GPT-4, Mistral, and LLaMA-2, to improve QDA tasks in RE. Our study evaluates LLMs' performance in inductive (zero-shot) and deductive (one-shot, few-shot) annotation tasks, revealing that GPT-4 achieves substantial agreement with human analysts in deductive settings, with Cohen's Kappa scores exceeding 0.7, while zero-shot performance remains limited. Detailed, context-rich prompts significantly improve annotation accuracy and consistency, particularly in deductive scenarios, and GPT-4 demonstrates high reliability across repeated runs. These findings highlight the potential of LLMs to support QDA in RE by reducing manual effort while maintaining annotation quality. The structured labels automatically provide traceability of requirements and can be directly utilized as classes in domain models, facilitating systematic software design. 

---
# CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges 

**Authors**: Yu Li, Qizhi Pei, Mengyuan Sun, Honglin Lin, Chenlin Ming, Xin Gao, Jiang Wu, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19093)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, especially the recent advancements in reasoning, such as o1 and o3, pushing the boundaries of AI. Despite these impressive achievements in mathematics and coding, the reasoning abilities of LLMs in domains requiring cryptographic expertise remain underexplored. In this paper, we introduce CipherBank, a comprehensive benchmark designed to evaluate the reasoning capabilities of LLMs in cryptographic decryption tasks. CipherBank comprises 2,358 meticulously crafted problems, covering 262 unique plaintexts across 5 domains and 14 subdomains, with a focus on privacy-sensitive and real-world scenarios that necessitate encryption. From a cryptographic perspective, CipherBank incorporates 3 major categories of encryption methods, spanning 9 distinct algorithms, ranging from classical ciphers to custom cryptographic techniques. We evaluate state-of-the-art LLMs on CipherBank, e.g., GPT-4o, DeepSeek-V3, and cutting-edge reasoning-focused models such as o1 and DeepSeek-R1. Our results reveal significant gaps in reasoning abilities not only between general-purpose chat LLMs and reasoning-focused LLMs but also in the performance of current reasoning-focused models when applied to classical cryptographic decryption tasks, highlighting the challenges these models face in understanding and manipulating encrypted data. Through detailed analysis and error investigations, we provide several key observations that shed light on the limitations and potential improvement areas for LLMs in cryptographic reasoning. These findings underscore the need for continuous advancements in LLM reasoning capabilities. 

---
# Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs 

**Authors**: Mohammad Akbar-Tajari, Mohammad Taher Pilehvar, Mohammad Mahmoody  

**Link**: [PDF](https://arxiv.org/pdf/2504.19019)  

**Abstract**: The challenge of ensuring Large Language Models (LLMs) align with societal standards is of increasing interest, as these models are still prone to adversarial jailbreaks that bypass their safety mechanisms. Identifying these vulnerabilities is crucial for enhancing the robustness of LLMs against such exploits. We propose Graph of ATtacks (GoAT), a method for generating adversarial prompts to test the robustness of LLM alignment using the Graph of Thoughts framework [Besta et al., 2024]. GoAT excels at generating highly effective jailbreak prompts with fewer queries to the victim model than state-of-the-art attacks, achieving up to five times better jailbreak success rate against robust models like Llama. Notably, GoAT creates high-quality, human-readable prompts without requiring access to the targeted model's parameters, making it a black-box attack. Unlike approaches constrained by tree-based reasoning, GoAT's reasoning is based on a more intricate graph structure. By making simultaneous attack paths aware of each other's progress, this dynamic framework allows a deeper integration and refinement of reasoning paths, significantly enhancing the collaborative exploration of adversarial vulnerabilities in LLMs. At a technical level, GoAT starts with a graph structure and iteratively refines it by combining and improving thoughts, enabling synergy between different thought paths. The code for our implementation can be found at: this https URL. 

---
# Why you shouldn't fully trust ChatGPT: A synthesis of this AI tool's error rates across disciplines and the software engineering lifecycle 

**Authors**: Vahid Garousi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18858)  

**Abstract**: Context: ChatGPT and other large language models (LLMs) are widely used across healthcare, business, economics, engineering, and software engineering (SE). Despite their popularity, concerns persist about their reliability, especially their error rates across domains and the software development lifecycle (SDLC).
Objective: This study synthesizes and quantifies ChatGPT's reported error rates across major domains and SE tasks aligned with SDLC phases. It provides an evidence-based view of where ChatGPT excels, where it fails, and how reliability varies by task, domain, and model version (GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o).
Method: A Multivocal Literature Review (MLR) was conducted, gathering data from academic studies, reports, benchmarks, and grey literature up to 2025. Factual, reasoning, coding, and interpretive errors were considered. Data were grouped by domain and SE phase and visualized using boxplots to show error distributions.
Results: Error rates vary across domains and versions. In healthcare, rates ranged from 8% to 83%. Business and economics saw error rates drop from ~50% with GPT-3.5 to 15-20% with GPT-4. Engineering tasks averaged 20-30%. Programming success reached 87.5%, though complex debugging still showed over 50% errors. In SE, requirements and design phases showed lower error rates (~5-20%), while coding, testing, and maintenance phases had higher variability (10-50%). Upgrades from GPT-3.5 to GPT-4 improved reliability.
Conclusion: Despite improvements, ChatGPT still exhibits non-negligible error rates varying by domain, task, and SDLC phase. Full reliance without human oversight remains risky, especially in critical settings. Continuous evaluation and critical validation are essential to ensure reliability and trustworthiness. 

---
# VeriDebug: A Unified LLM for Verilog Debugging via Contrastive Embedding and Guided Correction 

**Authors**: Ning Wang, Bingkun Yao, Jie Zhou, Yuchen Hu, Xi Wang, Nan Guan, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19099)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in debugging for various programming languages. However, the application of LLMs to Verilog debugging remains insufficiently explored. Here, we present VeriDebug, an approach that integrates contrastive representation and guided correction capabilities for automated Verilog debugging. Unlike existing methods, VeriDebug employs an embedding-based technique to accurately retrieve internal information, followed by bug-fixing. VeriDebug unifies Verilog bug detection and correction through a shared parameter space. By simultaneously learning bug patterns and fixes, it streamlines debugging via contrastive embedding and guided correction. Empirical results show the efficacy of VeriDebug in enhancing Verilog debugging. Our VeriDebugLoc, Type model achieves 64.7 accuracy in bug fixing (Acc1), a significant improvement from the existing open-source SOTAs 11.3. This performance not only outperforms open-source alternatives but also exceeds larger closed-source models like GPT-3.5-turbo (36.6), offering a more accurate alternative to conventional debugging methods. 

---
# TLoRA: Tri-Matrix Low-Rank Adaptation of Large Language Models 

**Authors**: Tanvir Islam  

**Link**: [PDF](https://arxiv.org/pdf/2504.18735)  

**Abstract**: We propose TLoRA, a novel tri-matrix low-rank adaptation method that decomposes weight updates into three matrices: two fixed random matrices and one trainable matrix, combined with a learnable, layer-wise scaling factor. This tri-matrix design enables TLoRA to achieve highly efficient parameter adaptation while introducing minimal additional computational overhead. Through extensive experiments on the GLUE benchmark, we demonstrate that TLoRA achieves comparable performance to existing low-rank methods such as LoRA and Adapter-based techniques, while requiring significantly fewer trainable parameters. Analyzing the adaptation dynamics, we observe that TLoRA exhibits Gaussian-like weight distributions, stable parameter norms, and scaling factor variability across layers, further highlighting its expressive power and adaptability. Additionally, we show that TLoRA closely resembles LoRA in its eigenvalue distributions, parameter norms, and cosine similarity of updates, underscoring its ability to effectively approximate LoRA's adaptation behavior. Our results establish TLoRA as a highly efficient and effective fine-tuning method for LLMs, offering a significant step forward in resource-efficient model adaptation. 

---
# MODP: Multi Objective Directional Prompting 

**Authors**: Aashutosh Nema, Samaksh Gulati, Evangelos Giakoumakis, Bipana Thapaliya  

**Link**: [PDF](https://arxiv.org/pdf/2504.18722)  

**Abstract**: Recent advances in large language models (LLMs) have led to their popularity across multiple use-cases. However, prompt engineering, the process for optimally utilizing such models, remains approximation-driven and subjective. Most of the current research on prompt engineering focuses on task-specific optimization, while neglecting the behavior of the LLM under consideration during prompt development. This paper introduces MODP -- Multi Objective Directional Prompting, a framework based on two key concepts: 1) multi-objectivity: the importance of considering an LLM's intrinsic behavior as an additional objective in prompt development, and 2) directional prompting: a metrics-driven method for prompt engineering to ensure development of robust and high-precision prompts. We demonstrate the effectiveness of our proposed ideas on a summarization task, using a synthetically created dataset, achieving a 26% performance gain over initial prompts. Finally, we apply MODP to develop prompts for Dell's Next Best Action support tool, which is now in production and is used by more than 10,000 internal support agents and serving millions of customers worldwide. 

---
# Test It Before You Trust It: Applying Software Testing for Trustworthy In-context Learning 

**Authors**: Teeradaj Racharak, Chaiyong Ragkhitwetsagul, Chommakorn Sontesadisai, Thanwadee Sunetnanta  

**Link**: [PDF](https://arxiv.org/pdf/2504.18827)  

**Abstract**: In-context learning (ICL) has emerged as a powerful capability of large language models (LLMs), enabling them to perform new tasks based on a few provided examples without explicit fine-tuning. Despite their impressive adaptability, these models remain vulnerable to subtle adversarial perturbations and exhibit unpredictable behavior when faced with linguistic variations. Inspired by software testing principles, we introduce a software testing-inspired framework, called MMT4NL, for evaluating the trustworthiness of in-context learning by utilizing adversarial perturbations and software testing techniques. It includes diverse evaluation aspects of linguistic capabilities for testing the ICL capabilities of LLMs. MMT4NL is built around the idea of crafting metamorphic adversarial examples from a test set in order to quantify and pinpoint bugs in the designed prompts of ICL. Our philosophy is to treat any LLM as software and validate its functionalities just like testing the software. Finally, we demonstrate applications of MMT4NL on the sentiment analysis and question-answering tasks. Our experiments could reveal various linguistic bugs in state-of-the-art LLMs. 

---
# SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models 

**Authors**: Nader Zantout, Haochen Zhang, Pujith Kachana, Jinkai Qiu, Ji Zhang, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18684)  

**Abstract**: Interpreting object-referential language and grounding objects in 3D with spatial relations and attributes is essential for robots operating alongside humans. However, this task is often challenging due to the diversity of scenes, large number of fine-grained objects, and complex free-form nature of language references. Furthermore, in the 3D domain, obtaining large amounts of natural language training data is difficult. Thus, it is important for methods to learn from little data and zero-shot generalize to new environments. To address these challenges, we propose SORT3D, an approach that utilizes rich object attributes from 2D data and merges a heuristics-based spatial reasoning toolbox with the ability of large language models (LLMs) to perform sequential reasoning. Importantly, our method does not require text-to-3D data for training and can be applied zero-shot to unseen environments. We show that SORT3D achieves state-of-the-art performance on complex view-dependent grounding tasks on two benchmarks. We also implement the pipeline to run real-time on an autonomous vehicle and demonstrate that our approach can be used for object-goal navigation on previously unseen real-world environments. All source code for the system pipeline is publicly released at this https URL . 

---
# Technical Challenges in Maintaining Tax Prep Software with Large Language Models 

**Authors**: Sina Gogani-Khiabani, Varsha Dewangan, Nina Olson, Ashutosh Trivedi, Saeid Tizpaz-Niari  

**Link**: [PDF](https://arxiv.org/pdf/2504.18693)  

**Abstract**: As the US tax law evolves to adapt to ever-changing politico-economic realities, tax preparation software plays a significant role in helping taxpayers navigate these complexities. The dynamic nature of tax regulations poses a significant challenge to accurately and timely maintaining tax software artifacts. The state-of-the-art in maintaining tax prep software is time-consuming and error-prone as it involves manual code analysis combined with an expert interpretation of tax law amendments. We posit that the rigor and formality of tax amendment language, as expressed in IRS publications, makes it amenable to automatic translation to executable specifications (code). Our research efforts focus on identifying, understanding, and tackling technical challenges in leveraging Large Language Models (LLMs), such as ChatGPT and Llama, to faithfully extract code differentials from IRS publications and automatically integrate them with the prior version of the code to automate tax prep software maintenance. 

---
# Can We Enhance Bug Report Quality Using LLMs?: An Empirical Study of LLM-Based Bug Report Generation 

**Authors**: Jagrit Acharya, Gouri Ginde  

**Link**: [PDF](https://arxiv.org/pdf/2504.18804)  

**Abstract**: Bug reports contain the information developers need to triage and fix software bugs. However, unclear, incomplete, or ambiguous information may lead to delays and excessive manual effort spent on bug triage and resolution. In this paper, we explore whether Instruction fine-tuned Large Language Models (LLMs) can automatically transform casual, unstructured bug reports into high-quality, structured bug reports adhering to a standard template. We evaluate three open-source instruction-tuned LLMs (\emph{Qwen 2.5, Mistral, and Llama 3.2}) against ChatGPT-4o, measuring performance on established metrics such as CTQRS, ROUGE, METEOR, and SBERT. Our experiments show that fine-tuned Qwen 2.5 achieves a CTQRS score of \textbf{77%}, outperforming both fine-tuned Mistral (\textbf{71%}), Llama 3.2 (\textbf{63%}) and ChatGPT in 3-shot learning (\textbf{75%}). Further analysis reveals that Llama 3.2 shows higher accuracy of detecting missing fields particularly Expected Behavior and Actual Behavior, while Qwen 2.5 demonstrates superior performance in capturing Steps-to-Reproduce, with an F1 score of 76%. Additional testing of the models on other popular projects (e.g., Eclipse, GCC) demonstrates that our approach generalizes well, achieving up to \textbf{70%} CTQRS in unseen projects' bug reports. These findings highlight the potential of instruction fine-tuning in automating structured bug report generation, reducing manual effort for developers and streamlining the software maintenance process. 

---
# Training Large Language Models to Reason via EM Policy Gradient 

**Authors**: Tianbing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18587)  

**Abstract**: Recently, foundation models such as OpenAI's O1 and O3, along with DeepSeek's R1, have demonstrated strong reasoning capacities and problem-solving skills acquired through large-scale reinforcement learning (RL), with wide applications in mathematics, coding, science, intelligent agents, and virtual assistants. In this work, we introduce an off-policy reinforcement learning algorithm, EM Policy Gradient, aimed at enhancing LLM reasoning by optimizing expected return over reasoning trajectories. We frame the reasoning task as an Expectation-Maximization (EM) optimization problem, alternating between sampling diverse rationale trajectories and performing reward-guided fine-tuning. Unlike PPO and GRPO, which rely on complex importance weights and heuristic clipping, our method provides a simpler, more principled off-policy policy gradient approach, eliminating these complexities while maintaining strong performance. We evaluate the effectiveness of EM Policy Gradient on the GSM8K and MATH (HARD) datasets, where it achieves performance comparable to or slightly surpassing the state-of-the-art GRPO, while offering additional advantages in scalability, simplicity, and reasoning conciseness. Moreover, models fine-tuned with our method exhibit cognitive behaviors, such as sub-problem decomposition, self-verification, and backtracking, highlighting its potential to enhance both the interpretability and robustness of LLM reasoning. 

---
# Toward Personalizing Quantum Computing Education: An Evolutionary LLM-Powered Approach 

**Authors**: Iizalaarab Elhaimeur, Nikos Chrisochoides  

**Link**: [PDF](https://arxiv.org/pdf/2504.18603)  

**Abstract**: Quantum computing education faces significant challenges due to its complexity and the limitations of current tools; this paper introduces a novel Intelligent Teaching Assistant for quantum computing education and details its evolutionary design process. The system combines a knowledge-graph-augmented architecture with two specialized Large Language Model (LLM) agents: a Teaching Agent for dynamic interaction, and a Lesson Planning Agent for lesson plan generation. The system is designed to adapt to individual student needs, with interactions meticulously tracked and stored in a knowledge graph. This graph represents student actions, learning resources, and relationships, aiming to enable reasoning about effective learning pathways. We describe the implementation of the system, highlighting the challenges encountered and the solutions implemented, including introducing a dual-agent architecture where tasks are separated, all coordinated through a central knowledge graph that maintains system awareness, and a user-facing tag system intended to mitigate LLM hallucination and improve user control. Preliminary results illustrate the system's potential to capture rich interaction data, dynamically adapt lesson plans based on student feedback via a tag system in simulation, and facilitate context-aware tutoring through the integrated knowledge graph, though systematic evaluation is required. 

---
# Large Language Model Empowered Privacy-Protected Framework for PHI Annotation in Clinical Notes 

**Authors**: Guanchen Wu, Linzhi Zheng, Han Xie, Zhen Xiang, Jiaying Lu, Darren Liu, Delgersuren Bold, Bo Li, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18569)  

**Abstract**: The de-identification of private information in medical data is a crucial process to mitigate the risk of confidentiality breaches, particularly when patient personal details are not adequately removed before the release of medical records. Although rule-based and learning-based methods have been proposed, they often struggle with limited generalizability and require substantial amounts of annotated data for effective performance. Recent advancements in large language models (LLMs) have shown significant promise in addressing these issues due to their superior language comprehension capabilities. However, LLMs present challenges, including potential privacy risks when using commercial LLM APIs and high computational costs for deploying open-source LLMs locally. In this work, we introduce LPPA, an LLM-empowered Privacy-Protected PHI Annotation framework for clinical notes, targeting the English language. By fine-tuning LLMs locally with synthetic notes, LPPA ensures strong privacy protection and high PHI annotation accuracy. Extensive experiments demonstrate LPPA's effectiveness in accurately de-identifying private information, offering a scalable and efficient solution for enhancing patient privacy protection. 

---
# DualBreach: Efficient Dual-Jailbreaking via Target-Driven Initialization and Multi-Target Optimization 

**Authors**: Xinzhe Huang, Kedong Xiu, Tianhang Zheng, Churui Zeng, Wangze Ni, Zhan Qiin, Kui Ren, Chun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18564)  

**Abstract**: Recent research has focused on exploring the vulnerabilities of Large Language Models (LLMs), aiming to elicit harmful and/or sensitive content from LLMs. However, due to the insufficient research on dual-jailbreaking -- attacks targeting both LLMs and Guardrails, the effectiveness of existing attacks is limited when attempting to bypass safety-aligned LLMs shielded by guardrails. Therefore, in this paper, we propose DualBreach, a target-driven framework for dual-jailbreaking. DualBreach employs a Target-driven Initialization (TDI) strategy to dynamically construct initial prompts, combined with a Multi-Target Optimization (MTO) method that utilizes approximate gradients to jointly adapt the prompts across guardrails and LLMs, which can simultaneously save the number of queries and achieve a high dual-jailbreaking success rate. For black-box guardrails, DualBreach either employs a powerful open-sourced guardrail or imitates the target black-box guardrail by training a proxy model, to incorporate guardrails into the MTO process.
We demonstrate the effectiveness of DualBreach in dual-jailbreaking scenarios through extensive evaluation on several widely-used datasets. Experimental results indicate that DualBreach outperforms state-of-the-art methods with fewer queries, achieving significantly higher success rates across all settings. More specifically, DualBreach achieves an average dual-jailbreaking success rate of 93.67% against GPT-4 with Llama-Guard-3 protection, whereas the best success rate achieved by other methods is 88.33%. Moreover, DualBreach only uses an average of 1.77 queries per successful dual-jailbreak, outperforming other state-of-the-art methods. For the purpose of defense, we propose an XGBoost-based ensemble defensive mechanism named EGuard, which integrates the strengths of multiple guardrails, demonstrating superior performance compared with Llama-Guard-3. 

---
