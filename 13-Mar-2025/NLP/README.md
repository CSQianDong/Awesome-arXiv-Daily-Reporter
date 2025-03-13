# MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System 

**Authors**: Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang, Simin Niu, Bo Tang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.09600)  

**Abstract**: Retrieval-Augmented Generation (RAG), while serving as a viable complement to large language models (LLMs), often overlooks the crucial aspect of text chunking within its pipeline. This paper initially introduces a dual-metric evaluation method, comprising Boundary Clarity and Chunk Stickiness, to enable the direct quantification of chunking quality. Leveraging this assessment method, we highlight the inherent limitations of traditional and semantic chunking in handling complex contextual nuances, thereby substantiating the necessity of integrating LLMs into chunking process. To address the inherent trade-off between computational efficiency and chunking precision in LLM-based approaches, we devise the granularity-aware Mixture-of-Chunkers (MoC) framework, which consists of a three-stage processing mechanism. Notably, our objective is to guide the chunker towards generating a structured list of chunking regular expressions, which are subsequently employed to extract chunks from the original text. Extensive experiments demonstrate that both our proposed metrics and the MoC framework effectively settle challenges of the chunking task, revealing the chunking kernel while enhancing the performance of the RAG system. 

---
# How to Protect Yourself from 5G Radiation? Investigating LLM Responses to Implicit Misinformation 

**Authors**: Ruohao Guo, Wei Xu, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2503.09598)  

**Abstract**: As Large Language Models (LLMs) are widely deployed in diverse scenarios, the extent to which they could tacitly spread misinformation emerges as a critical safety concern. Current research primarily evaluates LLMs on explicit false statements, overlooking how misinformation often manifests subtly as unchallenged premises in real-world user interactions. We curated ECHOMIST, the first comprehensive benchmark for implicit misinformation, where the misinformed assumptions are embedded in a user query to LLMs. ECHOMIST is based on rigorous selection criteria and carefully curated data from diverse sources, including real-world human-AI conversations and social media interactions. We also introduce a new evaluation metric to measure whether LLMs can recognize and counter false information rather than amplify users' misconceptions. Through an extensive empirical study on a wide range of LLMs, including GPT-4, Claude, and Llama, we find that current models perform alarmingly poorly on this task, often failing to detect false premises and generating misleading explanations. Our findings underscore the critical need for an increased focus on implicit misinformation in LLM safety research. 

---
# Cost-Optimal Grouped-Query Attention for Long-Context LLMs 

**Authors**: Yingfa Chen, Yutong Wu, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.09579)  

**Abstract**: Building effective and efficient Transformer-based large language models (LLMs) has recently become a research focus, requiring maximizing model language capabilities and minimizing training and deployment costs. Existing efforts have primarily described complex relationships among model performance, parameter size, and data size, as well as searched for the optimal compute allocation to train LLMs. However, they overlook the impacts of context length and attention head configuration (the number of query and key-value heads in grouped-query attention) on training and inference. In this paper, we systematically compare models with different parameter sizes, context lengths, and attention head configurations in terms of model performance, computational cost, and memory cost. Then, we extend the existing scaling methods, which are based solely on parameter size and training compute, to guide the construction of cost-optimal LLMs during both training and inference. Our quantitative scaling studies show that, when processing sufficiently long sequences, a larger model with fewer attention heads can achieve a lower loss while incurring lower computational and memory costs. Our findings provide valuable insights for developing practical LLMs, especially in long-context processing scenarios. We will publicly release our code and data. 

---
# Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks 

**Authors**: Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2503.09572)  

**Abstract**: Large language models (LLMs) have shown remarkable advancements in enabling language agents to tackle simple tasks. However, applying them for complex, multi-step, long-horizon tasks remains a challenge. Recent work have found success by separating high-level planning from low-level execution, which enables the model to effectively balance high-level planning objectives and low-level execution details. However, generating accurate plans remains difficult since LLMs are not inherently trained for this task. To address this, we propose Plan-and-Act, a novel framework that incorporates explicit planning into LLM-based agents and introduces a scalable method to enhance plan generation through a novel synthetic data generation method. Plan-and-Act consists of a Planner model which generates structured, high-level plans to achieve user goals, and an Executor model that translates these plans into environment-specific actions. To train the Planner effectively, we introduce a synthetic data generation method that annotates ground-truth trajectories with feasible plans, augmented with diverse and extensive examples to enhance generalization. We evaluate Plan-and-Act using web navigation as a representative long-horizon planning environment, demonstrating a state-of the-art 54% success rate on the WebArena-Lite benchmark. 

---
# PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs 

**Authors**: Oskar van der Wal, Pietro Lesci, Max Muller-Eberstein, Naomi Saphra, Hailey Schoelkopf, Willem Zuidema, Stella Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2503.09543)  

**Abstract**: The stability of language model pre-training and its effects on downstream performance are still understudied. Prior work shows that the training process can yield significantly different results in response to slight variations in initial conditions, e.g., the random seed. Crucially, the research community still lacks sufficient resources and tools to systematically investigate pre-training stability, particularly for decoder-only language models. We introduce the PolyPythias, a set of 45 new training runs for the Pythia model suite: 9 new seeds across 5 model sizes, from 14M to 410M parameters, resulting in about 7k new checkpoints that we release. Using these new 45 training runs, in addition to the 5 already available, we study the effects of different initial conditions determined by the seed -- i.e., parameters' initialisation and data order -- on (i) downstream performance, (ii) learned linguistic representations, and (iii) emergence of training phases. In addition to common scaling behaviours, our analyses generally reveal highly consistent training dynamics across both model sizes and initial conditions. Further, the new seeds for each model allow us to identify outlier training runs and delineate their characteristics. Our findings show the potential of using these methods to predict training stability. 

---
# Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning 

**Authors**: Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.09516)  

**Abstract**: Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Retrieval augmentation and tool-use training approaches where a search engine is treated as a tool lack complex multi-turn retrieval flexibility or require large-scale supervised data. Prompting advanced LLMs with reasoning capabilities during inference to use search engines is not optimal, since the LLM does not learn how to optimally interact with the search engine. This paper introduces Search-R1, an extension of the DeepSeek-R1 model where the LLM learns -- solely through reinforcement learning (RL) -- to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM rollouts with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 26% (Qwen2.5-7B), 21% (Qwen2.5-3B), and 10% (LLaMA3.2-3B) over SOTA baselines. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at this https URL. 

---
# TRACE: Real-Time Multimodal Common Ground Tracking in Situated Collaborative Dialogues 

**Authors**: Hannah VanderHoeven, Brady Bhalla, Ibrahim Khebour, Austin Youngren, Videep Venkatesha, Mariah Bradford, Jack Fitzgerald, Carlos Mabrey, Jingxuan Tu, Yifan Zhu, Kenneth Lai, Changsoo Jung, James Pustejovsky, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2503.09511)  

**Abstract**: We present TRACE, a novel system for live *common ground* tracking in situated collaborative tasks. With a focus on fast, real-time performance, TRACE tracks the speech, actions, gestures, and visual attention of participants, uses these multimodal inputs to determine the set of task-relevant propositions that have been raised as the dialogue progresses, and tracks the group's epistemic position and beliefs toward them as the task unfolds. Amid increased interest in AI systems that can mediate collaborations, TRACE represents an important step forward for agents that can engage with multiparty, multimodal discourse. 

---
# BAMBI: Developing Baby Language Models for Italian 

**Authors**: Alice Suozzi, Luca Capone, Gianluca E. Lebani, Alessandro Lenci  

**Link**: [PDF](https://arxiv.org/pdf/2503.09481)  

**Abstract**: This paper presents BAMBI (BAby language Models Boostrapped for Italian), a series of Baby Language Models (BabyLMs) trained on data that mimic the linguistic input received by a five-year-old Italian-speaking child. The BAMBI models are tested using a benchmark specifically designed to evaluate language models, which takes into account the amount of training input the models received. The BAMBI models are compared against a large language model (LLM) and a multimodal language model (VLM) to study the contribution of extralinguistic information for language acquisition. The results of our evaluation align with the existing literature on English language models, confirming that while reduced training data support the development of relatively robust syntactic competence, they are insufficient for fostering semantic understanding. However, the gap between the training resources (data and computation) of the BAMBI models and the LLMs is not fully reflected in their performance: despite LLMs' massive training, their performance is not much better than that of BAMBI models. This suggests that strategies beyond scaling training resources, such as data curation, inclusion of multimodal input, and other training strategies such as curriculum learning, could play a crucial role in shaping model performance. 

---
# Explicit Learning and the LLM in Machine Translation 

**Authors**: Malik Marmonier, Rachel Bawden, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2503.09454)  

**Abstract**: This study explores the capacity of large language models (LLMs) for explicit learning, a process involving the assimilation of metalinguistic explanations to carry out language tasks. Using constructed languages generated by cryptographic means as controlled test environments, we designed experiments to assess an LLM's ability to explicitly learn and apply grammar rules. Our results demonstrate that while LLMs possess a measurable capacity for explicit learning, this ability diminishes as the complexity of the linguistic phenomena at hand increases. Supervised fine-tuning on chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs. 

---
# Florenz: Scaling Laws for Systematic Generalization in Vision-Language Models 

**Authors**: Julian Spravil, Sebastian Houben, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2503.09443)  

**Abstract**: Cross-lingual transfer enables vision-language models (VLMs) to perform vision tasks in various languages with training data only in one language. Current approaches rely on large pre-trained multilingual language models. However, they face the curse of multilinguality, sacrificing downstream task performance for multilingual capabilities, struggling with lexical ambiguities, and falling behind recent advances. In this work, we study the scaling laws of systematic generalization with monolingual VLMs for multilingual tasks, focusing on the impact of model size and seen training samples. We propose Florenz, a monolingual encoder-decoder VLM with 0.4B to 11.2B parameters combining the pre-trained VLM Florence-2 and the large language model Gemma-2. Florenz is trained with varying compute budgets on a synthetic dataset that features intentionally incomplete language coverage for image captioning, thus, testing generalization from the fully covered translation task. We show that not only does indirectly learning unseen task-language pairs adhere to a scaling law, but also that with our data generation pipeline and the proposed Florenz model family, image captioning abilities can emerge in a specific language even when only data for the translation task is available. Fine-tuning on a mix of downstream datasets yields competitive performance and demonstrates promising scaling trends in multimodal machine translation (Multi30K, CoMMuTE), lexical disambiguation (CoMMuTE), and image captioning (Multi30K, XM3600, COCO Karpathy). 

---
# Towards Generating Automatic Anaphora Annotations 

**Authors**: Dima Taji, Daniel Zeman  

**Link**: [PDF](https://arxiv.org/pdf/2503.09417)  

**Abstract**: Training models that can perform well on various NLP tasks require large amounts of data, and this becomes more apparent with nuanced tasks such as anaphora and conference resolution. To combat the prohibitive costs of creating manual gold annotated data, this paper explores two methods to automatically create datasets with coreferential annotations; direct conversion from existing datasets, and parsing using multilingual models capable of handling new and unseen languages. The paper details the current progress on those two fronts, as well as the challenges the efforts currently face, and our approach to overcoming these challenges. 

---
# Got Compute, but No Data: Lessons From Post-training a Finnish LLM 

**Authors**: Elaine Zosa, Ville Komulainen, Sampo Pyysalo  

**Link**: [PDF](https://arxiv.org/pdf/2503.09407)  

**Abstract**: As LLMs gain more popularity as chatbots and general assistants, methods have been developed to enable LLMs to follow instructions and align with human preferences. These methods have found success in the field, but their effectiveness has not been demonstrated outside of high-resource languages. In this work, we discuss our experiences in post-training an LLM for instruction-following for English and Finnish. We use a multilingual LLM to translate instruction and preference datasets from English to Finnish. We perform instruction tuning and preference optimization in English and Finnish and evaluate the instruction-following capabilities of the model in both languages. Our results show that with a few hundred Finnish instruction samples we can obtain competitive performance in Finnish instruction-following. We also found that although preference optimization in English offers some cross-lingual benefits, we obtain our best results by using preference data from both languages. We release our model, datasets, and recipes under open licenses at this https URL 

---
# RetSTA: An LLM-Based Approach for Standardizing Clinical Fundus Image Reports 

**Authors**: Jiushen Cai, Weihang Zhang, Hanruo Liu, Ningli Wang, Huiqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.09358)  

**Abstract**: Standardization of clinical reports is crucial for improving the quality of healthcare and facilitating data integration. The lack of unified standards, including format, terminology, and style, is a great challenge in clinical fundus diagnostic reports, which increases the difficulty for large language models (LLMs) to understand the data. To address this, we construct a bilingual standard terminology, containing fundus clinical terms and commonly used descriptions in clinical diagnosis. Then, we establish two models, RetSTA-7B-Zero and RetSTA-7B. RetSTA-7B-Zero, fine-tuned on an augmented dataset simulating clinical scenarios, demonstrates powerful standardization behaviors. However, it encounters a challenge of limitation to cover a wider range of diseases. To further enhance standardization performance, we build RetSTA-7B, which integrates a substantial amount of standardized data generated by RetSTA-7B-Zero along with corresponding English data, covering diverse complex clinical scenarios and achieving report-level standardization for the first time. Experimental results demonstrate that RetSTA-7B outperforms other compared LLMs in bilingual standardization task, which validates its superior performance and generalizability. The checkpoints are available at this https URL. 

---
# MOAT: Evaluating LMMs for Capability Integration and Instruction Grounding 

**Authors**: Zhoutong Ye, Mingze Sun, Huan-ang Gao, Chun Yu, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09348)  

**Abstract**: Large multimodal models (LMMs) have demonstrated significant potential as generalists in vision-language (VL) tasks. However, there remains a significant gap between state-of-the-art LMMs and human performance when it comes to complex tasks that require a combination of fundamental VL capabilities, as well as tasks involving the grounding of complex instructions. To thoroughly investigate the human-LMM gap and its underlying causes, we propose MOAT, a diverse benchmark with complex real-world VL tasks that are challenging for LMMs. Specifically, the tasks in MOAT require LMMs to engage in generalist problem solving by integrating fundamental VL capabilities such as reading text, counting, understanding spatial relations, grounding textual and visual instructions, etc. All these abilities fit into a taxonomy proposed by us that contains 10 fundamental VL capabilities, enabling MOAT to provide a fine-grained view of LMMs' strengths and weaknesses. Besides, MOAT is the first benchmark to explicitly evaluate LMMs' ability to ground complex text and visual instructions, which is essential to many real-world applications. We evaluate over 20 proprietary and open source LMMs, as well as humans, on MOAT, and found that humans achieved 82.7% accuracy while the best performing LMM (OpenAI o1) achieved only 38.8%. To guide future model development, we analyze common trends in our results and discuss the underlying causes of observed performance gaps between LMMs and humans, focusing on which VL capability forms the bottleneck in complex tasks, whether test time scaling improves performance on MOAT, and how tiling harms LMMs' capability to count. Code and data are available at this https URL. 

---
# Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts 

**Authors**: Hongyu Chen, Seraphina Goldfarb-Tarrant  

**Link**: [PDF](https://arxiv.org/pdf/2503.09347)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments. 

---
# An Evaluation of LLMs for Detecting Harmful Computing Terms 

**Authors**: Joshua Jacas, Hana Winchester, Alicia Boyd, Brittany Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2503.09341)  

**Abstract**: Detecting harmful and non-inclusive terminology in technical contexts is critical for fostering inclusive environments in computing. This study explores the impact of model architecture on harmful language detection by evaluating a curated database of technical terms, each paired with specific use cases. We tested a range of encoder, decoder, and encoder-decoder language models, including BERT-base-uncased, RoBERTa large-mnli, Gemini Flash 1.5 and 2.0, GPT-4, Claude AI Sonnet 3.5, T5-large, and BART-large-mnli. Each model was presented with a standardized prompt to identify harmful and non-inclusive language across 64 terms. Results reveal that decoder models, particularly Gemini Flash 2.0 and Claude AI, excel in nuanced contextual analysis, while encoder models like BERT exhibit strong pattern recognition but struggle with classification certainty. We discuss the implications of these findings for improving automated detection tools and highlight model-specific strengths and limitations in fostering inclusive communication in technical domains. 

---
# Investigating User Perspectives on Differentially Private Text Privatization 

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Alexander Karpp, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2503.09338)  

**Abstract**: Recent literature has seen a considerable uptick in $\textit{Differentially Private Natural Language Processing}$ (DP NLP). This includes DP text privatization, where potentially sensitive input texts are transformed under DP to achieve privatized output texts that ideally mask sensitive information $\textit{and}$ maintain original semantics. Despite continued work to address the open challenges in DP text privatization, there remains a scarcity of work addressing user perceptions of this technology, a crucial aspect which serves as the final barrier to practical adoption. In this work, we conduct a survey study with 721 laypersons around the globe, investigating how the factors of $\textit{scenario}$, $\textit{data sensitivity}$, $\textit{mechanism type}$, and $\textit{reason for data collection}$ impact user preferences for text privatization. We learn that while all these factors play a role in influencing privacy decisions, users are highly sensitive to the utility and coherence of the private output texts. Our findings highlight the socio-technical factors that must be considered in the study of DP NLP, opening the door to further user-based investigations going forward. 

---
# A Survey on Enhancing Causal Reasoning Ability of Large Language Models 

**Authors**: Xin Li, Zhuo Cai, Shoujin Wang, Kun Yu, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.09326)  

**Abstract**: Large language models (LLMs) have recently shown remarkable performance in language tasks and beyond. However, due to their limited inherent causal reasoning ability, LLMs still face challenges in handling tasks that require robust causal reasoning ability, such as health-care and economic analysis. As a result, a growing body of research has focused on enhancing the causal reasoning ability of LLMs. Despite the booming research, there lacks a survey to well review the challenges, progress and future directions in this area. To bridge this significant gap, we systematically review literature on how to strengthen LLMs' causal reasoning ability in this paper. We start from the introduction of background and motivations of this topic, followed by the summarisation of key challenges in this area. Thereafter, we propose a novel taxonomy to systematically categorise existing methods, together with detailed comparisons within and between classes of methods. Furthermore, we summarise existing benchmarks and evaluation metrics for assessing LLMs' causal reasoning ability. Finally, we outline future research directions for this emerging field, offering insights and inspiration to researchers and practitioners in the area. 

---
# xVLM2Vec: Adapting LVLM-based embedding models to multilinguality using Self-Knowledge Distillation 

**Authors**: Elio Musacchio, Lucia Siciliani, Pierpaolo Basile, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2503.09313)  

**Abstract**: In the current literature, most embedding models are based on the encoder-only transformer architecture to extract a dense and meaningful representation of the given input, which can be a text, an image, and more. With the recent advances in language modeling thanks to the introduction of Large Language Models, the possibility of extracting embeddings from these large and extensively trained models has been explored. However, current studies focus on textual embeddings in English, which is also the main language on which these models have been trained. Furthermore, there are very few models that consider multimodal and multilingual input. In light of this, we propose an adaptation methodology for Large Vision-Language Models trained on English language data to improve their performance in extracting multilingual and multimodal embeddings. Finally, we design and introduce a benchmark to evaluate the effectiveness of multilingual and multimodal embedding models. 

---
# Unmask It! AI-Generated Product Review Detection in Dravidian Languages 

**Authors**: Somsubhra De, Advait Vats  

**Link**: [PDF](https://arxiv.org/pdf/2503.09289)  

**Abstract**: The rise of Generative AI has led to a surge in AI-generated reviews, often posing a serious threat to the credibility of online platforms. Reviews serve as the primary source of information about products and services. Authentic reviews play a vital role in consumer decision-making. The presence of fabricated content misleads consumers, undermines trust and facilitates potential fraud in digital marketplaces. This study focuses on detecting AI-generated product reviews in Tamil and Malayalam, two low-resource languages where research in this domain is relatively under-explored. We worked on a range of approaches - from traditional machine learning methods to advanced transformer-based models such as Indic-BERT, IndicSBERT, MuRIL, XLM-RoBERTa and MalayalamBERT. Our findings highlight the effectiveness of leveraging the state-of-the-art transformers in accurately identifying AI-generated content, demonstrating the potential in enhancing the detection of fake reviews in low-resource language settings. 

---
# Considering Length Diversity in Retrieval-Augmented Summarization 

**Authors**: Juseon-Do, Jaesung Hwang, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2503.09249)  

**Abstract**: This study investigates retrieval-augmented summarization by specifically examining the impact of exemplar summary lengths under length constraints, not covered by previous work. We propose a Diverse Length-aware Maximal Marginal Relevance (DL-MMR) algorithm to better control summary lengths. This algorithm combines the query relevance with diverse target lengths in retrieval-augmented summarization. Unlike previous methods that necessitate exhaustive exemplar exemplar relevance comparisons using MMR, DL-MMR considers the exemplar target length as well and avoids comparing exemplars to each other, thereby reducing computational cost and conserving memory during the construction of an exemplar pool. Experimental results showed the effectiveness of DL-MMR, which considers length diversity, compared to the original MMR algorithm. DL-MMR additionally showed the effectiveness in memory saving of 781,513 times and computational cost reduction of 500,092 times, while maintaining the same level of informativeness. 

---
# Rethinking Prompt-based Debiasing in Large Language Models 

**Authors**: Xinyi Yang, Runzhe Zhan, Derek F. Wong, Shu Yang, Junchao Wu, Lidia S. Chao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09219)  

**Abstract**: Investigating bias in large language models (LLMs) is crucial for developing trustworthy AI. While prompt-based through prompt engineering is common, its effectiveness relies on the assumption that models inherently understand biases. Our study systematically analyzed this assumption using the BBQ and StereoSet benchmarks on both open-source models as well as commercial GPT model. Experimental results indicate that prompt-based is often superficial; for instance, the Llama2-7B-Chat model misclassified over 90% of unbiased content as biased, despite achieving high accuracy in identifying bias issues on the BBQ dataset. Additionally, specific evaluation and question settings in bias benchmarks often lead LLMs to choose "evasive answers", disregarding the core of the question and the relevance of the response to the context. Moreover, the apparent success of previous methods may stem from flawed evaluation metrics. Our research highlights a potential "false prosperity" in prompt-base efforts and emphasizes the need to rethink bias metrics to ensure truly trustworthy AI. 

---
# N2C2: Nearest Neighbor Enhanced Confidence Calibration for Cross-Lingual In-Context Learning 

**Authors**: Jie He, Simon Yu, Deyi Xiong, Víctor Gutiérrez-Basulto, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.09218)  

**Abstract**: Recent advancements of in-context learning (ICL) show language models can significantly improve their performance when demonstrations are provided. However, little attention has been paid to model calibration and prediction confidence of ICL in cross-lingual scenarios. To bridge this gap, we conduct a thorough analysis of ICL for cross-lingual sentiment classification. Our findings suggest that ICL performs poorly in cross-lingual scenarios, exhibiting low accuracy and presenting high calibration errors. In response, we propose a novel approach, N2C2, which employs a -nearest neighbors augmented classifier for prediction confidence calibration. N2C2 narrows the prediction gap by leveraging a datastore of cached few-shot instances. Specifically, N2C2 integrates the predictions from the datastore and incorporates confidence-aware distribution, semantically consistent retrieval representation, and adaptive neighbor combination modules to effectively utilize the limited number of supporting instances. Evaluation on two multilingual sentiment classification datasets demonstrates that N2C2 outperforms traditional ICL. It surpasses fine tuning, prompt tuning and recent state-of-the-art methods in terms of accuracy and calibration errors. 

---
# Token Weighting for Long-Range Language Modeling 

**Authors**: Falko Helm, Nico Daheim, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.09202)  

**Abstract**: Many applications of large language models (LLMs) require long-context understanding, but models continue to struggle with such tasks. We hypothesize that conventional next-token prediction training could contribute to this, because each token is assigned equal weight. Yet, intuitively, the amount of context needed to predict the next token accurately varies greatly across different data. To reflect this, we propose various novel token-weighting schemes that assign different weights to each training token in the loss, thereby generalizing existing works. For this, we categorize token-weighting methods using a two-step framework which compares the confidences of a long-context and short-context model to score tokens. We evaluate all methods on multiple long-context understanding tasks and show that non-uniform loss weights are helpful to improve the long-context abilities of LLMs. Different short-context models can be used effectively for token scoring, including models that are much smaller than the long-context model that is trained. All in all, this work contributes to a better understanding of the trade-offs long-context language modeling faces and provides guidelines for model steering via loss-weighting based on empirical evidence. The code can be found on Github. 

---
# Is LLMs Hallucination Usable? LLM-based Negative Reasoning for Fake News Detection 

**Authors**: Chaowei Zhang, Zongling Feng, Zewei Zhang, Jipeng Qiang, Guandong Xu, Yun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.09153)  

**Abstract**: The questionable responses caused by knowledge hallucination may lead to LLMs' unstable ability in decision-making. However, it has never been investigated whether the LLMs' hallucination is possibly usable to generate negative reasoning for facilitating the detection of fake news. This study proposes a novel supervised self-reinforced reasoning rectification approach - SR$^3$ that yields both common reasonable reasoning and wrong understandings (negative reasoning) for news via LLMs reflection for semantic consistency learning. Upon that, we construct a negative reasoning-based news learning model called - \emph{NRFE}, which leverages positive or negative news-reasoning pairs for learning the semantic consistency between them. To avoid the impact of label-implicated reasoning, we deploy a student model - \emph{NRFE-D} that only takes news content as input to inspect the performance of our method by distilling the knowledge from \emph{NRFE}. The experimental results verified on three popular fake news datasets demonstrate the superiority of our method compared with three kinds of baselines including prompting on LLMs, fine-tuning on pre-trained SLMs, and other representative fake news detection methods. 

---
# VaxGuard: A Multi-Generator, Multi-Type, and Multi-Role Dataset for Detecting LLM-Generated Vaccine Misinformation 

**Authors**: Syed Talal Ahmad, Haohui Lu, Sidong Liu, Annie Lau, Amin Beheshti, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2503.09103)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly improved text generation capabilities. However, they also present challenges, particularly in generating vaccine-related misinformation, which poses risks to public health. Despite research on human-authored misinformation, a notable gap remains in understanding how LLMs contribute to vaccine misinformation and how best to detect it. Existing benchmarks often overlook vaccine-specific misinformation and the diverse roles of misinformation spreaders. This paper introduces VaxGuard, a novel dataset designed to address these challenges. VaxGuard includes vaccine-related misinformation generated by multiple LLMs and provides a comprehensive framework for detecting misinformation across various roles. Our findings show that GPT-3.5 and GPT-4o consistently outperform other LLMs in detecting misinformation, especially when dealing with subtle or emotionally charged narratives. On the other hand, PHI3 and Mistral show lower performance, struggling with precision and recall in fear-driven contexts. Additionally, detection performance tends to decline as input text length increases, indicating the need for improved methods to handle larger content. These results highlight the importance of role-specific detection strategies and suggest that VaxGuard can serve as a key resource for improving the detection of LLM-generated vaccine misinformation. 

---
# Domain Adaptation for Japanese Sentence Embeddings with Contrastive Learning based on Synthetic Sentence Generation 

**Authors**: Zihao Chen, Hisashi Handa, Miho Ohsaki, Kimiaki Shirahama  

**Link**: [PDF](https://arxiv.org/pdf/2503.09094)  

**Abstract**: Several backbone models pre-trained on general domain datasets can encode a sentence into a widely useful embedding. Such sentence embeddings can be further enhanced by domain adaptation that adapts a backbone model to a specific domain. However, domain adaptation for low-resource languages like Japanese is often difficult due to the scarcity of large-scale labeled datasets. To overcome this, this paper introduces SDJC (Self-supervised Domain adaptation for Japanese sentence embeddings with Contrastive learning) that utilizes a data generator to generate sentences, which have the same syntactic structure to a sentence in an unlabeled specific domain corpus but convey different semantic meanings. Generated sentences are then used to boost contrastive learning that adapts a backbone model to accurately discriminate sentences in the specific domain. In addition, the components of SDJC like a backbone model and a method to adapt it need to be carefully selected, but no benchmark dataset is available for Japanese. Thus, a comprehensive Japanese STS (Semantic Textual Similarity) benchmark dataset is constructed by combining datasets machine-translated from English with existing datasets. The experimental results validates the effectiveness of SDJC on two domain-specific downstream tasks as well as the usefulness of the constructed dataset. Datasets, codes and backbone models adapted by SDJC are available on our github repository this https URL. 

---
# DAST: Difficulty-Aware Self-Training on Large Language Models 

**Authors**: Boyang Xue, Qi Zhu, Hongru Wang, Rui Wang, Sheng Wang, Hongling Xu, Fei Mi, Yasheng Wang, Lifeng Shang, Qun Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09029)  

**Abstract**: Present Large Language Models (LLM) self-training methods always under-sample on challenging queries, leading to inadequate learning on difficult problems which limits LLMs' ability. Therefore, this work proposes a difficulty-aware self-training (DAST) framework that focuses on improving both the quantity and quality of self-generated responses on challenging queries during self-training. DAST is specified in three components: 1) sampling-based difficulty level estimation, 2) difficulty-aware data augmentation, and 3) the self-training algorithm using SFT and DPO respectively. Experiments on mathematical tasks demonstrate the effectiveness and generalization of DAST, highlighting the critical role of difficulty-aware strategies in advancing LLM self-training. 

---
# Aligning to What? Limits to RLHF Based Alignment 

**Authors**: Logan Barnhart, Reza Akbarian Bafghi, Stephen Becker, Maziar Raissi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09025)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is increasingly used to align large language models (LLMs) with human preferences. However, the effectiveness of RLHF in addressing underlying biases remains unclear. This study investigates the relationship between RLHF and both covert and overt biases in LLMs, particularly focusing on biases against African Americans. We applied various RLHF techniques (DPO, ORPO, and RLOO) to Llama 3 8B and evaluated the covert and overt biases of the resulting models using matched-guise probing and explicit bias testing. We performed additional tests with DPO on different base models and datasets; among several implications, we found that SFT before RLHF calcifies model biases. Additionally, we extend the tools for measuring biases to multi-modal models. Through our experiments we collect evidence that indicates that current alignment techniques are inadequate for nebulous tasks such as mitigating covert biases, highlighting the need for capable datasets, data curating techniques, or alignment tools. 

---
# Word2winners at SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval 

**Authors**: Amirmohammad Azadi, Sina Zamani, Mohammadmostafa Rostamkhani, Sauleh Eetemadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09011)  

**Abstract**: This paper describes our system for SemEval 2025 Task 7: Previously Fact-Checked Claim Retrieval. The task requires retrieving relevant fact-checks for a given input claim from the extensive, multilingual MultiClaim dataset, which comprises social media posts and fact-checks in several languages. To address this challenge, we first evaluated zero-shot performance using state-of-the-art English and multilingual retrieval models and then fine-tuned the most promising systems, leveraging machine translation to enhance crosslingual retrieval. Our best model achieved an accuracy of 85% on crosslingual data and 92% on monolingual data. 

---
# Agentic AI for Scientific Discovery: A Survey of Progress, Challenges, and Future Directions 

**Authors**: Mourad Gridach, Jay Nanavati, Khaldoun Zine El Abidine, Lenon Mendes, Christina Mack  

**Link**: [PDF](https://arxiv.org/pdf/2503.08979)  

**Abstract**: The integration of Agentic AI into scientific discovery marks a new frontier in research automation. These AI systems, capable of reasoning, planning, and autonomous decision-making, are transforming how scientists perform literature review, generate hypotheses, conduct experiments, and analyze results. This survey provides a comprehensive overview of Agentic AI for scientific discovery, categorizing existing systems and tools, and highlighting recent progress across fields such as chemistry, biology, and materials science. We discuss key evaluation metrics, implementation frameworks, and commonly used datasets to offer a detailed understanding of the current state of the field. Finally, we address critical challenges, such as literature review automation, system reliability, and ethical concerns, while outlining future research directions that emphasize human-AI collaboration and enhanced system calibration. 

---
# Gradient-guided Attention Map Editing: Towards Efficient Contextual Hallucination Mitigation 

**Authors**: Yu Wang, Jiaxin Zhang, Xiang Gao, Wendi Cui, Peng Li, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.08963)  

**Abstract**: In tasks like summarization and open-book question answering (QA), Large Language Models (LLMs) often encounter "contextual hallucination", where they produce irrelevant or incorrect responses despite having access to accurate source information. This typically occurs because these models tend to prioritize self-generated content over the input context, causing them to disregard pertinent details. To address this challenge, we introduce a novel method called "Guided Attention Map Editing" (GAME), which dynamically adjusts attention maps to improve contextual relevance. During inference, GAME employs a trained classifier to identify attention maps prone to inducing hallucinations and executes targeted interventions. These interventions, guided by gradient-informed "edit directions'', strategically redistribute attention weights across various heads to effectively reduce hallucination. Comprehensive evaluations on challenging summarization and open-book QA tasks show that GAME consistently reduces hallucinations across a variety of open-source models. Specifically, GAME reduces hallucinations by 10% in the XSum summarization task while achieving a 7X speed-up in computational efficiency compared to the state-of-the-art baselines. 

---
# Backtracking for Safety 

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda  

**Link**: [PDF](https://arxiv.org/pdf/2503.08919)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency. 

---
# EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees 

**Authors**: Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, Pang Wei Koh  

**Link**: [PDF](https://arxiv.org/pdf/2503.08893)  

**Abstract**: An ideal model evaluation should achieve two goals: identifying where the model fails and providing actionable improvement guidance. Toward these goals for Language Model (LM) evaluations, we formulate the problem of generating a weakness profile, a set of weaknesses expressed in natural language, given an LM's performance on every individual instance in a benchmark. We introduce a suite of quantitative assessments to compare different weakness profiling methods. We also propose a weakness profiling method EvalTree. It constructs a capability tree where each node represents a capability described in natural language and is linked to a subset of benchmark instances that specifically evaluate this capability; it then extracts nodes where the LM performs poorly to generate a weakness profile. On the MATH and WildChat benchmarks, we show that EvalTree outperforms baseline weakness profiling methods by identifying weaknesses more precisely and comprehensively. Weakness profiling further enables weakness-guided data collection, and training data collection guided by EvalTree-identified weaknesses improves LM performance more than other data collection strategies. We also show how EvalTree exposes flaws in Chatbot Arena's human-voter-based evaluation practice. To facilitate future work, we release our code and an interface that allows practitioners to interactively explore the capability trees built by EvalTree. 

---
# PlainQAFact: Automatic Factuality Evaluation Metric for Biomedical Plain Language Summaries Generation 

**Authors**: Zhiwen You, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08890)  

**Abstract**: Hallucinated outputs from language models pose risks in the medical domain, especially for lay audiences making health-related decisions. Existing factuality evaluation methods, such as entailment- and question-answering-based (QA), struggle with plain language summary (PLS) generation due to elaborative explanation phenomenon, which introduces external content (e.g., definitions, background, examples) absent from the source document to enhance comprehension. To address this, we introduce PlainQAFact, a framework trained on a fine-grained, human-annotated dataset PlainFact, to evaluate the factuality of both source-simplified and elaboratively explained sentences. PlainQAFact first classifies factuality type and then assesses factuality using a retrieval-augmented QA-based scoring method. Our approach is lightweight and computationally efficient. Empirical results show that existing factuality metrics fail to effectively evaluate factuality in PLS, especially for elaborative explanations, whereas PlainQAFact achieves state-of-the-art performance. We further analyze its effectiveness across external knowledge sources, answer extraction strategies, overlap measures, and document granularity levels, refining its overall factuality assessment. 

---
# LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference 

**Authors**: Guangtao Wang, Shubhangi Upasani, Chen Wu, Darshan Gandhi, Jonathan Li, Changran Hu, Bo Li, Urmish Thakker  

**Link**: [PDF](https://arxiv.org/pdf/2503.08879)  

**Abstract**: Efficient long-context inference is critical as large language models (LLMs) adopt context windows of ranging from 128K to 1M tokens. However, the growing key-value (KV) cache and the high computational complexity of attention create significant bottlenecks in memory usage and latency. In this paper, we find that attention in diverse long-context tasks exhibits sparsity, and LLMs implicitly "know" which tokens can be dropped or evicted at the head level after the pre-filling stage. Based on this insight, we propose Self-Attention Guided Eviction~(SAGE-KV), a simple and effective KV eviction cache method for long-context inference. After prefilling, our method performs a one-time top-k selection at both the token and head levels to compress the KV cache, enabling efficient inference with the reduced cache. Evaluations on LongBench and three long-context LLMs (Llama3.1-8B-Instruct-128k, Llama3-8B-Prolong-512k-Instruct, and Qwen2.5-7B-Instruct-128k) show that SAGE-KV maintains accuracy comparable to full attention while significantly improving efficiency. Specifically, SAGE-KV achieves 4x higher memory efficiency with improved accuracy over the static KV cache selection method StreamLLM, and 2x higher memory efficiency with better accuracy than the dynamic KV cache selection method Quest. 

---
# Interpretable and Robust Dialogue State Tracking via Natural Language Summarization with LLMs 

**Authors**: Rafael Carranza, Mateo Alejandro Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2503.08857)  

**Abstract**: This paper introduces a novel approach to Dialogue State Tracking (DST) that leverages Large Language Models (LLMs) to generate natural language descriptions of dialogue states, moving beyond traditional slot-value representations. Conventional DST methods struggle with open-domain dialogues and noisy inputs. Motivated by the generative capabilities of LLMs, our Natural Language DST (NL-DST) framework trains an LLM to directly synthesize human-readable state descriptions. We demonstrate through extensive experiments on MultiWOZ 2.1 and Taskmaster-1 datasets that NL-DST significantly outperforms rule-based and discriminative BERT-based DST baselines, as well as generative slot-filling GPT-2 DST models, in both Joint Goal Accuracy and Slot Accuracy. Ablation studies and human evaluations further validate the effectiveness of natural language state generation, highlighting its robustness to noise and enhanced interpretability. Our findings suggest that NL-DST offers a more flexible, accurate, and human-understandable approach to dialogue state tracking, paving the way for more robust and adaptable task-oriented dialogue systems. 

---
# Contrastive Speaker-Aware Learning for Multi-party Dialogue Generation with LLMs 

**Authors**: Tianyu Sun, Kun Qian, Wenhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08842)  

**Abstract**: Multi-party dialogue generation presents significant challenges due to the complex interplay of multiple speakers and interwoven conversational threads. Traditional approaches often fall short in capturing these complexities, particularly when relying on manually annotated dialogue relations. This paper introduces Speaker-Attentive LLM (SA-LLM), a novel generative model that leverages pre-trained Large Language Models (LLMs) and a speaker-aware contrastive learning strategy to address these challenges. SA-LLM incorporates a speaker-attributed input encoding and a contrastive learning objective to implicitly learn contextual coherence and speaker roles without explicit relation annotations. Extensive experiments on the Ubuntu IRC and Movie Dialogues datasets demonstrate that SA-LLM significantly outperforms state-of-the-art baselines in automatic and human evaluations, achieving superior performance in fluency, coherence, informativeness, and response diversity. Ablation studies and detailed error analyses further validate the effectiveness of the proposed speaker-attentive training approach, highlighting its robustness across different speaker roles and context lengths. The results underscore the potential of SA-LLM as a powerful and annotation-free solution for high-quality multi-party dialogue generation. 

---
# evoBPE: Evolutionary Protein Sequence Tokenization 

**Authors**: Burak Suyunu, Özdeniz Dolu, Arzucan Özgür  

**Link**: [PDF](https://arxiv.org/pdf/2503.08838)  

**Abstract**: Recent advancements in computational biology have drawn compelling parallels between protein sequences and linguistic structures, highlighting the need for sophisticated tokenization methods that capture the intricate evolutionary dynamics of protein sequences. Current subword tokenization techniques, primarily developed for natural language processing, often fail to represent protein sequences' complex structural and functional properties adequately. This study introduces evoBPE, a novel tokenization approach that integrates evolutionary mutation patterns into sequence segmentation, addressing critical limitations in existing methods. By leveraging established substitution matrices, evoBPE transcends traditional frequency-based tokenization strategies. The method generates candidate token pairs through biologically informed mutations, evaluating them based on pairwise alignment scores and frequency thresholds. Extensive experiments on human protein sequences show that evoBPE performs better across multiple dimensions. Domain conservation analysis reveals that evoBPE consistently outperforms standard Byte-Pair Encoding, particularly as vocabulary size increases. Furthermore, embedding similarity analysis using ESM-2 suggests that mutation-based token replacements preserve biological sequence properties more effectively than arbitrary substitutions. The research contributes to protein sequence representation by introducing a mutation-aware tokenization method that better captures evolutionary nuances. By bridging computational linguistics and molecular biology, evoBPE opens new possibilities for machine learning applications in protein function prediction, structural modeling, and evolutionary analysis. 

---
# Cross-Examiner: Evaluating Consistency of Large Language Model-Generated Explanations 

**Authors**: Danielle Villa, Maria Chang, Keerthiram Murugesan, Rosario Uceda-Sosa, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08815)  

**Abstract**: Large Language Models (LLMs) are often asked to explain their outputs to enhance accuracy and transparency. However, evidence suggests that these explanations can misrepresent the models' true reasoning processes. One effective way to identify inaccuracies or omissions in these explanations is through consistency checking, which typically involves asking follow-up questions. This paper introduces, cross-examiner, a new method for generating follow-up questions based on a model's explanation of an initial question. Our method combines symbolic information extraction with language model-driven question generation, resulting in better follow-up questions than those produced by LLMs alone. Additionally, this approach is more flexible than other methods and can generate a wider variety of follow-up questions. 

---
# ESNLIR: A Spanish Multi-Genre Dataset with Causal Relationships 

**Authors**: Johan R. Portela, Nicolás Perez, Rubén Manrique  

**Link**: [PDF](https://arxiv.org/pdf/2503.08803)  

**Abstract**: Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), serves as a crucial area within the domain of Natural Language Processing (NLP). This area fundamentally empowers machines to discern semantic relationships between assorted sections of text. Even though considerable work has been executed for the English language, it has been observed that efforts for the Spanish language are relatively sparse. Keeping this in view, this paper focuses on generating a multi-genre Spanish dataset for NLI, ESNLIR, particularly accounting for causal Relationships. A preliminary baseline has been conceptualized and subjected to an evaluation, leveraging models drawn from the BERT family. The findings signify that the enrichment of genres essentially contributes to the enrichment of the model's capability to generalize.
The code, notebooks and whole datasets for this experiments is available at: this https URL. If you are interested only in the dataset you can find it here: this https URL. 

---
# Exposing Product Bias in LLM Investment Recommendation 

**Authors**: Yuhan Zhi, Xiaoyu Zhang, Longtian Wang, Shumin Jiang, Shiqing Ma, Xiaohong Guan, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08750)  

**Abstract**: Large language models (LLMs), as a new generation of recommendation engines, possess powerful summarization and data analysis capabilities, surpassing traditional recommendation systems in both scope and performance. One promising application is investment recommendation. In this paper, we reveal a novel product bias in LLM investment recommendation, where LLMs exhibit systematic preferences for specific products. Such preferences can subtly influence user investment decisions, potentially leading to inflated valuations of products and financial bubbles, posing risks to both individual investors and market stability. To comprehensively study the product bias, we develop an automated pipeline to create a dataset of 567,000 samples across five asset classes (stocks, mutual funds, cryptocurrencies, savings, and portfolios). With this dataset, we present the bf first study on product bias in LLM investment recommendations. Our findings reveal that LLMs exhibit clear product preferences, such as certain stocks (e.g., `AAPL' from Apple and `MSFT' from Microsoft). Notably, this bias persists even after applying debiasing techniques. We urge AI researchers to take heed of the product bias in LLM investment recommendations and its implications, ensuring fairness and security in the digital space and market. 

---
# Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, Wangxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2503.09567)  

**Abstract**: Recent advancements in reasoning with large language models (RLLMs), such as OpenAI-O1 and DeepSeek-R1, have demonstrated their impressive capabilities in complex domains like mathematics and coding. A central factor in their success lies in the application of long chain-of-thought (Long CoT) characteristics, which enhance reasoning abilities and enable the solution of intricate problems. However, despite these developments, a comprehensive survey on Long CoT is still lacking, limiting our understanding of its distinctions from traditional short chain-of-thought (Short CoT) and complicating ongoing debates on issues like "overthinking" and "test-time scaling." This survey seeks to fill this gap by offering a unified perspective on Long CoT. (1) We first distinguish Long CoT from Short CoT and introduce a novel taxonomy to categorize current reasoning paradigms. (2) Next, we explore the key characteristics of Long CoT: deep reasoning, extensive exploration, and feasible reflection, which enable models to handle more complex tasks and produce more efficient, coherent outcomes compared to the shallower Short CoT. (3) We then investigate key phenomena such as the emergence of Long CoT with these characteristics, including overthinking, and test-time scaling, offering insights into how these processes manifest in practice. (4) Finally, we identify significant research gaps and highlight promising future directions, including the integration of multi-modal reasoning, efficiency improvements, and enhanced knowledge frameworks. By providing a structured overview, this survey aims to inspire future research and further the development of logical reasoning in artificial intelligence. 

---
# SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability 

**Authors**: Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall, Kola Ayonrinde, Matthew Wearden, Arthur Conmy, Samuel Marks, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2503.09532)  

**Abstract**: Sparse autoencoders (SAEs) are a popular technique for interpreting language model activations, and there is extensive recent work on improving SAE effectiveness. However, most prior work evaluates progress using unsupervised proxy metrics with unclear practical relevance. We introduce SAEBench, a comprehensive evaluation suite that measures SAE performance across seven diverse metrics, spanning interpretability, feature disentanglement and practical applications like unlearning. To enable systematic comparison, we open-source a suite of over 200 SAEs across eight recently proposed SAE architectures and training algorithms. Our evaluation reveals that gains on proxy metrics do not reliably translate to better practical performance. For instance, while Matryoshka SAEs slightly underperform on existing proxy metrics, they substantially outperform other architectures on feature disentanglement metrics; moreover, this advantage grows with SAE scale. By providing a standardized framework for measuring progress in SAE development, SAEBench enables researchers to study scaling trends and make nuanced comparisons between different SAE architectures and training methodologies. Our interactive interface enables researchers to flexibly visualize relationships between metrics across hundreds of open-source SAEs at: this https URL 

---
# Reinforcement Learning is all You Need 

**Authors**: Yongsheng Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.09512)  

**Abstract**: Inspired by the success of DeepSeek R1 in reasoning via reinforcement learning without human feedback, we train a 3B language model using the Countdown Game with pure reinforcement learning. Our model outperforms baselines on four of five benchmarks, demonstrating improved generalization beyond its training data. Notably, response length does not correlate with reasoning quality, and while "aha moments" emerge, they do not always yield correct answers. These findings highlight the potential of RL-only training for reasoning enhancement and suggest future work on refining reward structures to bridge emergent insights with accuracy. 

---
# ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning 

**Authors**: Ziyu Wan, Yunxiang Li, Yan Song, Hanjing Wang, Linyi Yang, Mark Schmidt, Jun Wang, Weinan Zhang, Shuyue Hu, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.09501)  

**Abstract**: Recent research on Reasoning of Large Language Models (LLMs) has sought to further enhance their performance by integrating meta-thinking -- enabling models to monitor, evaluate, and control their reasoning processes for more adaptive and effective problem-solving. However, current single-agent work lacks a specialized design for acquiring meta-thinking, resulting in low efficacy. To address this challenge, we introduce Reinforced Meta-thinking Agents (ReMA), a novel framework that leverages Multi-Agent Reinforcement Learning (MARL) to elicit meta-thinking behaviors, encouraging LLMs to think about thinking. ReMA decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions. Through iterative reinforcement learning with aligned objectives, these agents explore and learn collaboration, leading to improved generalization and robustness. Experimental results demonstrate that ReMA outperforms single-agent RL baselines on complex reasoning tasks, including competitive-level mathematical benchmarks and LLM-as-a-Judge benchmarks. Comprehensive ablation studies further illustrate the evolving dynamics of each distinct agent, providing valuable insights into how the meta-thinking reasoning process enhances the reasoning capabilities of LLMs. 

---
# MindGYM: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions 

**Authors**: Zhe Xu, Daoyuan Chen, Zhenqing Ling, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.09499)  

**Abstract**: Large vision-language models (VLMs) face challenges in achieving robust, transferable reasoning abilities due to reliance on labor-intensive manual instruction datasets or computationally expensive self-supervised methods. To address these issues, we introduce MindGYM, a framework that enhances VLMs through synthetic self-challenging questions, consisting of three stages: (1) Seed Single-Hop Question Synthesis, generating cognitive questions across textual (e.g., logical deduction) and multimodal contexts (e.g., diagram-based queries) spanning eight semantic areas like ethical analysis; (2) Challenging Multi-Hop Question Synthesis, combining seed questions via diverse principles like bridging, visual-textual alignment, to create multi-step problems demanding deeper reasoning; and (3) Thinking-Induced Curriculum Fine-Tuning, a structured pipeline that progressively trains the model from scaffolded reasoning to standalone inference. By leveraging the model's self-synthesis capability, MindGYM achieves high data efficiency (e.g., +16% gains on MathVision-Mini with only 400 samples), computational efficiency (reducing both training and inference costs), and robust generalization across tasks. Extensive evaluations on seven benchmarks demonstrate superior performance over strong baselines, with notable improvements (+15.77% win rates) in reasoning depth and breadth validated via GPT-based scoring. MindGYM underscores the viability of self-challenging for refining VLM capabilities while minimizing human intervention and resource demands. Code and data are released to advance multimodal reasoning research. 

---
# Why LLMs Cannot Think and How to Fix It 

**Authors**: Marius Jahrens, Thomas Martinetz  

**Link**: [PDF](https://arxiv.org/pdf/2503.09211)  

**Abstract**: This paper elucidates that current state-of-the-art Large Language Models (LLMs) are fundamentally incapable of making decisions or developing "thoughts" within the feature space due to their architectural constraints. We establish a definition of "thought" that encompasses traditional understandings of that term and adapt it for application to LLMs. We demonstrate that the architectural design and language modeling training methodology of contemporary LLMs inherently preclude them from engaging in genuine thought processes. Our primary focus is on this theoretical realization rather than practical insights derived from experimental data. Finally, we propose solutions to enable thought processes within the feature space and discuss the broader implications of these architectural modifications. 

---
# Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model 

**Authors**: Ali Vosoughi, Dimitra Emmanouilidou, Hannes Gamper  

**Link**: [PDF](https://arxiv.org/pdf/2503.09205)  

**Abstract**: Integrating audio and visual data for training multimodal foundational models remains challenging. We present Audio-Video Vector Alignment (AVVA), which aligns audiovisual (AV) scene content beyond mere temporal synchronization via a Large Language Model (LLM)-based data curation pipeline. Specifically, AVVA scores and selects high-quality training clips using Whisper (speech-based audio foundation model) for audio and DINOv2 for video within a dual-encoder contrastive learning framework. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate that this approach can achieve significant accuracy gains with substantially less curated data. For instance, AVVA yields a 7.6% improvement in top-1 accuracy for audio-to-video retrieval on VGGSound compared to ImageBind, despite training on only 192 hours of carefully filtered data (vs. 5800+ hours). Moreover, an ablation study highlights that trading data quantity for data quality improves performance, yielding respective top-3 accuracy increases of 47.8, 48.4, and 58.0 percentage points on AudioCaps, VALOR, and VGGSound over uncurated baselines. While these results underscore AVVA's data efficiency, we also discuss the overhead of LLM-driven curation and how it may be scaled or approximated in larger domains. Overall, AVVA provides a viable path toward more robust, text-free audiovisual learning with improved retrieval accuracy. 

---
# Specification languages for computational laws versus basic legal principles 

**Authors**: Petia Guintchev, Joost J. Joosten, Sofia Santiago Fernández, Eric Sancho Adamson, Aleix Solé Sánchez, Marta Soria Heredia  

**Link**: [PDF](https://arxiv.org/pdf/2503.09129)  

**Abstract**: We speak of a \textit{computational law} when that law is intended to be enforced by software through an automated decision-making process. As digital technologies evolve to offer more solutions for public administrations, we see an ever-increasing number of computational laws. Traditionally, law is written in natural language. Computational laws, however, suffer various complications when written in natural language, such as underspecification and ambiguity which lead to a diversity of possible interpretations to be made by the coder. These could potentially result into an uneven application of the law. Thus, resorting to formal languages to write computational laws is tempting. However, writing laws in a formal language leads to further complications, for example, incomprehensibility for non-experts, lack of explicit motivation of the decisions made, or difficulties in retrieving the data leading to the outcome. In this paper, we investigate how certain legal principles fare in both scenarios: computational law written in natural language or written in formal language. We use a running example from the European Union's road transport regulation to showcase the tensions arising, and the benefits from each language. 

---
# GRU: Mitigating the Trade-off between Unlearning and Retention for Large Language Models 

**Authors**: Yue Wang, Qizhou Wang, Feng Liu, Wei Huang, Yali Du, Xiaojiang Du, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.09117)  

**Abstract**: Large language model (LLM) unlearning has demonstrated its essential role in removing privacy and copyright-related responses, crucial for their legal and safe applications. However, the pursuit of complete unlearning often comes with substantial costs due to its compromises in their general functionality, leading to a notorious trade-off between unlearning and retention. In examining the update process for unlearning dynamically, we find gradients hold essential information for revealing this trade-off. In particular, we look at the varying relationship between retention performance and directional disparities between gradients during unlearning. It motivates the sculpting of an update mechanism derived from gradients from two sources, i.e., harmful for retention and useful for unlearning. Accordingly, we propose Gradient Rectified Unlearning (GRU), an enhanced unlearning framework controlling the updating gradients in a geometry-focused and optimization-driven manner such that their side impacts on other, unrelated responses can be minimized. Specifically, GRU derives a closed-form solution to project the unlearning gradient onto the orthogonal space of that gradient harmful for retention, ensuring minimal deviation from its original direction under the condition that overall performance is retained. Comprehensive experiments are conducted to demonstrate that GRU, as a general framework, is straightforward to implement and efficiently enhances a range of baseline methods through its adaptable and compatible characteristics. Additionally, experimental results show its broad effectiveness across a diverse set of benchmarks for LLM unlearning. 

---
# LocAgent: Graph-Guided LLM Agents for Code Localization 

**Authors**: Zhaoling Chen, Xiangru Tang, Gangda Deng, Fang Wu, Jialong Wu, Zhiwei Jiang, Viktor Prasanna, Arman Cohan, Xingyao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09089)  

**Abstract**: Code localization--identifying precisely where in a codebase changes need to be made--is a fundamental yet challenging task in software maintenance. Existing approaches struggle to efficiently navigate complex codebases when identifying relevant code sections. The challenge lies in bridging natural language problem descriptions with the appropriate code elements, often requiring reasoning across hierarchical structures and multiple dependencies. We introduce LocAgent, a framework that addresses code localization through graph-based representation. By parsing codebases into directed heterogeneous graphs, LocAgent creates a lightweight representation that captures code structures (files, classes, functions) and their dependencies (imports, invocations, inheritance), enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning. Experimental results on real-world benchmarks demonstrate that our approach significantly enhances accuracy in code localization. Notably, our method with the fine-tuned Qwen-2.5-Coder-Instruct-32B model achieves comparable results to SOTA proprietary models at greatly reduced cost (approximately 86% reduction), reaching up to 92.7% accuracy on file-level localization while improving downstream GitHub issue resolution success rates by 12% for multiple attempts (Pass@10). Our code is available at this https URL. 

---
# Teaching LLMs How to Learn with Contextual Fine-Tuning 

**Authors**: Younwoo Choi, Muhammad Adil Asif, Ziwen Han, John Willes, Rahul G. Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2503.09032)  

**Abstract**: Prompting Large Language Models (LLMs), or providing context on the expected model of operation, is an effective way to steer the outputs of such models to satisfy human desiderata after they have been trained. But in rapidly evolving domains, there is often need to fine-tune LLMs to improve either the kind of knowledge in their memory or their abilities to perform open ended reasoning in new domains. When human's learn new concepts, we often do so by linking the new material that we are studying to concepts we have already learned before. To that end, we ask, "can prompting help us teach LLMs how to learn". In this work, we study a novel generalization of instruction tuning, called contextual fine-tuning, to fine-tune LLMs. Our method leverages instructional prompts designed to mimic human cognitive strategies in learning and problem-solving to guide the learning process during training, aiming to improve the model's interpretation and understanding of domain-specific knowledge. We empirically demonstrate that this simple yet effective modification improves the ability of LLMs to be fine-tuned rapidly on new datasets both within the medical and financial domains. 

---
# Leveraging Retrieval Augmented Generative LLMs For Automated Metadata Description Generation to Enhance Data Catalogs 

**Authors**: Mayank Singh, Abhijeet Kumar, Sasidhar Donaparthi, Gayatri Karambelkar  

**Link**: [PDF](https://arxiv.org/pdf/2503.09003)  

**Abstract**: Data catalogs serve as repositories for organizing and accessing diverse collection of data assets, but their effectiveness hinges on the ease with which business users can look-up relevant content. Unfortunately, many data catalogs within organizations suffer from limited searchability due to inadequate metadata like asset descriptions. Hence, there is a need of content generation solution to enrich and curate metadata in a scalable way. This paper explores the challenges associated with metadata creation and proposes a unique prompt enrichment idea of leveraging existing metadata content using retrieval based few-shot technique tied with generative large language models (LLM). The literature also considers finetuning an LLM on existing content and studies the behavior of few-shot pretrained LLM (Llama, GPT3.5) vis-à-vis few-shot finetuned LLM (Llama2-7b) by evaluating their performance based on accuracy, factual grounding, and toxicity. Our preliminary results exhibit more than 80% Rouge-1 F1 for the generated content. This implied 87%- 88% of instances accepted as is or curated with minor edits by data stewards. By automatically generating descriptions for tables and columns in most accurate way, the research attempts to provide an overall framework for enterprises to effectively scale metadata curation and enrich its data catalog thereby vastly improving the data catalog searchability and overall usability. 

---
# JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing 

**Authors**: Vasudev Gohil  

**Link**: [PDF](https://arxiv.org/pdf/2503.08990)  

**Abstract**: Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs.
JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers. 

---
# I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data? 

**Authors**: Yuhang Liu, Dong Gong, Erdun Gao, Zhen Zhang, Biwei Huang, Mingming Gong, Anton van den Hengel, Javen Qinfeng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08980)  

**Abstract**: The remarkable achievements of large language models (LLMs) have led many to conclude that they exhibit a form of intelligence. This is as opposed to explanations of their capabilities based on their ability to perform relatively simple manipulations of vast volumes of data. To illuminate the distinction between these explanations, we introduce a novel generative model that generates tokens on the basis of human interpretable concepts represented as latent discrete variables. Under mild conditions, even when the mapping from the latent space to the observed space is non-invertible, we establish an identifiability result: the representations learned by LLMs through next-token prediction can be approximately modeled as the logarithm of the posterior probabilities of these latent discrete concepts, up to an invertible linear transformation. This theoretical finding not only provides evidence that LLMs capture underlying generative factors, but also strongly reinforces the linear representation hypothesis, which posits that LLMs learn linear representations of human-interpretable concepts. Empirically, we validate our theoretical results through evaluations on both simulation data and the Pythia, Llama, and DeepSeek model families. 

---
# An Exhaustive Evaluation of TTS- and VC-based Data Augmentation for ASR 

**Authors**: Sewade Ogun, Vincent Colotte, Emmanuel Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2503.08954)  

**Abstract**: Augmenting the training data of automatic speech recognition (ASR) systems with synthetic data generated by text-to-speech (TTS) or voice conversion (VC) has gained popularity in recent years. Several works have demonstrated improvements in ASR performance using this augmentation approach. However, because of the lower diversity of synthetic speech, naively combining synthetic and real data often does not yield the best results. In this work, we leverage recently proposed flow-based TTS/VC models allowing greater speech diversity, and assess the respective impact of augmenting various speech attributes on the word error rate (WER) achieved by several ASR models. Pitch augmentation and VC-based speaker augmentation are found to be ineffective in our setup. Jointly augmenting all other attributes reduces the WER of a Conformer-Transducer model by 11\% relative on Common Voice and by up to 35\% relative on LibriSpeech compared to training on real data only. 

---
# Interpreting the Repeated Token Phenomenon in Large Language Models 

**Authors**: Itay Yona, Ilia Shumailov, Jamie Hayes, Federico Barbero, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2503.08908)  

**Abstract**: Large Language Models (LLMs), despite their impressive capabilities, often fail to accurately repeat a single word when prompted to, and instead output unrelated text. This unexplained failure mode represents a vulnerability, allowing even end-users to diverge models away from their intended behavior. We aim to explain the causes for this phenomenon and link it to the concept of ``attention sinks'', an emergent LLM behavior crucial for fluency, in which the initial token receives disproportionately high attention scores. Our investigation identifies the neural circuit responsible for attention sinks and shows how long repetitions disrupt this circuit. We extend this finding to other non-repeating sequences that exhibit similar circuit disruptions. To address this, we propose a targeted patch that effectively resolves the issue without negatively impacting the model's overall performance. This study provides a mechanistic explanation for an LLM vulnerability, demonstrating how interpretability can diagnose and address issues, and offering insights that pave the way for more secure and reliable models. 

---
# Prompt-OT: An Optimal Transport Regularization Paradigm for Knowledge Preservation in Vision-Language Model Adaptation 

**Authors**: Xiwen Chen, Wenhui Zhu, Peijie Qiu, Hao Wang, Huayu Li, Haiyu Wu, Aristeidis Sotiras, Yalin Wang, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08906)  

**Abstract**: Vision-language models (VLMs) such as CLIP demonstrate strong performance but struggle when adapted to downstream tasks. Prompt learning has emerged as an efficient and effective strategy to adapt VLMs while preserving their pre-trained knowledge. However, existing methods still lead to overfitting and degrade zero-shot generalization. To address this challenge, we propose an optimal transport (OT)-guided prompt learning framework that mitigates forgetting by preserving the structural consistency of feature distributions between pre-trained and fine-tuned models. Unlike conventional point-wise constraints, OT naturally captures cross-instance relationships and expands the feasible parameter space for prompt tuning, allowing a better trade-off between adaptation and generalization. Our approach enforces joint constraints on both vision and text representations, ensuring a holistic feature alignment. Extensive experiments on benchmark datasets demonstrate that our simple yet effective method can outperform existing prompt learning strategies in base-to-novel generalization, cross-dataset evaluation, and domain generalization without additional augmentation or ensemble techniques. The code is available at this https URL 

---
# Seeing What's Not There: Spurious Correlation in Multimodal LLMs 

**Authors**: Parsa Hosseini, Sumit Nawathe, Mazda Moayeri, Sriram Balasubramanian, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08884)  

**Abstract**: Unimodal vision models are known to rely on spurious correlations, but it remains unclear to what extent Multimodal Large Language Models (MLLMs) exhibit similar biases despite language supervision. In this paper, we investigate spurious bias in MLLMs and introduce SpurLens, a pipeline that leverages GPT-4 and open-set object detectors to automatically identify spurious visual cues without human supervision. Our findings reveal that spurious correlations cause two major failure modes in MLLMs: (1) over-reliance on spurious cues for object recognition, where removing these cues reduces accuracy, and (2) object hallucination, where spurious cues amplify the hallucination by over 10x. We validate our findings in various MLLMs and datasets. Beyond diagnosing these failures, we explore potential mitigation strategies, such as prompt ensembling and reasoning-based prompting, and conduct ablation studies to examine the root causes of spurious bias in MLLMs. By exposing the persistence of spurious correlations, our study calls for more rigorous evaluation methods and mitigation strategies to enhance the reliability of MLLMs. 

---
# ResBench: Benchmarking LLM-Generated FPGA Designs with Resource Awareness 

**Authors**: Ce Guo, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08823)  

**Abstract**: Field-Programmable Gate Arrays (FPGAs) are widely used in modern hardware design, yet writing Hardware Description Language (HDL) code for FPGA implementation remains labor-intensive and complex. Large Language Models (LLMs) have emerged as a promising tool for automating HDL generation, but existing benchmarks for LLM HDL code generation primarily evaluate functional correctness while overlooking the critical aspect of hardware resource efficiency. Moreover, current benchmarks lack diversity, failing to capture the broad range of real-world FPGA applications. To address these gaps, we introduce ResBench, the first resource-oriented benchmark explicitly designed to differentiate between resource-optimized and inefficient LLM-generated HDL. ResBench consists of 56 problems across 12 categories, covering applications from finite state machines to financial computing. Our evaluation framework systematically integrates FPGA resource constraints, with a primary focus on Lookup Table (LUT) usage, enabling a realistic assessment of hardware efficiency. Experimental results reveal substantial differences in resource utilization across LLMs, demonstrating ResBench's effectiveness in distinguishing models based on their ability to generate resource-optimized FPGA designs. 

---
