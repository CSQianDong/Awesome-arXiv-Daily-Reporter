# LCES: Zero-shot Automated Essay Scoring via Pairwise Comparisons Using Large Language Models 

**Authors**: Takumi Shibata, Yuichi Miyamura  

**Link**: [PDF](https://arxiv.org/pdf/2505.08498)  

**Abstract**: Recent advances in large language models (LLMs) have enabled zero-shot automated essay scoring (AES), providing a promising way to reduce the cost and effort of essay scoring in comparison with manual grading. However, most existing zero-shot approaches rely on LLMs to directly generate absolute scores, which often diverge from human evaluations owing to model biases and inconsistent scoring. To address these limitations, we propose LLM-based Comparative Essay Scoring (LCES), a method that formulates AES as a pairwise comparison task. Specifically, we instruct LLMs to judge which of two essays is better, collect many such comparisons, and convert them into continuous scores. Considering that the number of possible comparisons grows quadratically with the number of essays, we improve scalability by employing RankNet to efficiently transform LLM preferences into scalar scores. Experiments using AES benchmark datasets show that LCES outperforms conventional zero-shot methods in accuracy while maintaining computational efficiency. Moreover, LCES is robust across different LLM backbones, highlighting its applicability to real-world zero-shot AES. 

---
# Are LLMs complicated ethical dilemma analyzers? 

**Authors**: Jiashen, Jesse Yao, Allen Liu, Zhekai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08106)  

**Abstract**: One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making. 

---
# Efficient Telecom Specific LLM: TSLAM-Mini with QLoRA and Digital Twin Data 

**Authors**: Vignesh Ethiraj, Divya Vijay, Sidhanth Menon, Heblin Berscilla  

**Link**: [PDF](https://arxiv.org/pdf/2505.07877)  

**Abstract**: General-purpose large language models (LLMs), despite their broad capabilities accrued from open-world data, frequently exhibit suboptimal performance when confronted with the nuanced and specialized demands inherent in real-time telecommunications applications. This investigation addresses this critical limitation through the meticulous fine-tuning of TSLAM-Mini developed by NetoAI, a compact (3.8-billion parameter) causal language model architecturally derived from Phi-4 Mini Instruct 4B. The fine-tuning regimen leverages a bespoke dataset comprising 100,000 samples, strategically engineered to address 20 pivotal telecommunications use-cases, encompassing domains such as Network Fundamentals, IP Routing, MPLS, Network Security, Automation, OSS/BSS, RAN, Mobile Core, Satellite Communications, and Ethical AI. This dataset was curated utilizing NetoAI's DigiTwin platform, enriched with granular insights from venerated network Subject Matter Experts (SMEs) and authoritative RFC documents, thereby capturing high-fidelity representations of real-world network dynamics through simulations inspired by digital twin paradigms. Employing Quantized Low-Rank Adaptation (QLoRA), a state-of-the-art Parameter Efficient Fine-Tuning (PEFT) technique, we achieved substantial training efficiency and enabled prospective deployment on resource-constrained hardware. A novel evaluation framework, predicated on a high-capacity LLM (Qwen3-235B-A22B) functioning as an automated adjudicator, was instituted to rigorously assess instruction-following fidelity and response quality across the specified telecom use-cases. Empirical results unequivocally demonstrate TSLAM-Mini's superior aptitude in telecom-centric applications, underscoring the profound efficacy of domain-specific datasets and PEFT methodologies for advancing intelligent network management. 

---
# Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment 

**Authors**: Ali Senol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07852)  

**Abstract**: Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models. 

---
# Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning? 

**Authors**: Md Tahmid Rahman Laskar, Mohammed Saidul Islam, Ridwan Mahbub, Ahmed Masry, Mizanur Rahman, Amran Bhuiyan, Mir Tafseer Nayeem, Shafiq Joty, Enamul Hoque, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08468)  

**Abstract**: Charts are ubiquitous as they help people understand and reason with data. Recently, various downstream tasks, such as chart question answering, chart2text, and fact-checking, have emerged. Large Vision-Language Models (LVLMs) show promise in tackling these tasks, but their evaluation is costly and time-consuming, limiting real-world deployment. While using LVLMs as judges to assess the chart comprehension capabilities of other LVLMs could streamline evaluation processes, challenges like proprietary datasets, restricted access to powerful models, and evaluation costs hinder their adoption in industrial settings. To this end, we present a comprehensive evaluation of 13 open-source LVLMs as judges for diverse chart comprehension and reasoning tasks. We design both pairwise and pointwise evaluation tasks covering criteria like factual correctness, informativeness, and relevancy. Additionally, we analyze LVLM judges based on format adherence, positional consistency, length bias, and instruction-following. We focus on cost-effective LVLMs (<10B parameters) suitable for both research and commercial use, following a standardized evaluation protocol and rubric to measure the LVLM judge's accuracy. Experimental results reveal notable variability: while some open LVLM judges achieve GPT-4-level evaluation performance (about 80% agreement with GPT-4 judgments), others struggle (below ~10% agreement). Our findings highlight that state-of-the-art open-source LVLMs can serve as cost-effective automatic evaluators for chart-related tasks, though biases such as positional preference and length bias persist. 

---
