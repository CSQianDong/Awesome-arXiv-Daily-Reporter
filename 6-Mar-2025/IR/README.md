# Addressing Overprescribing Challenges: Fine-Tuning Large Language Models for Medication Recommendation Tasks 

**Authors**: Zihao Zhao, Chenxiao Fan, Chongming Gao, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.03687)  

**Abstract**: Medication recommendation systems have garnered attention within healthcare for their potential to deliver personalized and efficacious drug combinations based on patient's clinical data. However, existing methodologies encounter challenges in adapting to diverse Electronic Health Records (EHR) systems and effectively utilizing unstructured data, resulting in limited generalization capabilities and suboptimal performance. Recently, interest is growing in harnessing Large Language Models (LLMs) in the medical domain to support healthcare professionals and enhance patient care. Despite the emergence of medical LLMs and their promising results in tasks like medical question answering, their practical applicability in clinical settings, particularly in medication recommendation, often remains underexplored.
In this study, we evaluate both general-purpose and medical-specific LLMs for medication recommendation tasks. Our findings reveal that LLMs frequently encounter the challenge of overprescribing, leading to heightened clinical risks and diminished medication recommendation accuracy. To address this issue, we propose Language-Assisted Medication Recommendation (LAMO), which employs a parameter-efficient fine-tuning approach to tailor open-source LLMs for optimal performance in medication recommendation scenarios. LAMO leverages the wealth of clinical information within clinical notes, a resource often underutilized in traditional methodologies. As a result of our approach, LAMO outperforms previous state-of-the-art methods by over 10% in internal validation accuracy. Furthermore, temporal and external validations demonstrate LAMO's robust generalization capabilities across various temporal and hospital contexts. Additionally, an out-of-distribution medication recommendation experiment demonstrates LAMO's remarkable accuracy even with medications outside the training data. 

---
# Decoupled Recommender Systems: Exploring Alternative Recommender Ecosystem Designs 

**Authors**: Anas Buhayh, Elizabeth McKinnie, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2503.03606)  

**Abstract**: Recommender ecosystems are an emerging subject of research. Such research examines how the characteristics of algorithms, recommendation consumers, and item providers influence system dynamics and long-term outcomes. One architectural possibility that has not yet been widely explored in this line of research is the consequences of a configuration in which recommendation algorithms are decoupled from the platforms they serve. This is sometimes called "the friendly neighborhood algorithm store" or "middleware" model. We are particularly interested in how such architectures might offer a range of different distributions of utility across consumers, providers, and recommendation platforms. In this paper, we create a model of a recommendation ecosystem that incorporates algorithm choice and examine the outcomes of such a design. 

---
# Intrinsic and Extrinsic Factor Disentanglement for Recommendation in Various Context Scenarios 

**Authors**: Yixin Su, Wei Jiang, Fangquan Lin, Cheng Yang, Sarah M. Erfani, Junhao Gan, Yunxiang Zhao, Ruixuan Li, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03524)  

**Abstract**: In recommender systems, the patterns of user behaviors (e.g., purchase, click) may vary greatly in different contexts (e.g., time and location). This is because user behavior is jointly determined by two types of factors: intrinsic factors, which reflect consistent user preference, and extrinsic factors, which reflect external incentives that may vary in different contexts. Differentiating between intrinsic and extrinsic factors helps learn user behaviors better. However, existing studies have only considered differentiating them from a single, pre-defined context (e.g., time or location), ignoring the fact that a user's extrinsic factors may be influenced by the interplay of various contexts at the same time. In this paper, we propose the Intrinsic-Extrinsic Disentangled Recommendation (IEDR) model, a generic framework that differentiates intrinsic from extrinsic factors considering various contexts simultaneously, enabling more accurate differentiation of factors and hence the improvement of recommendation accuracy. IEDR contains a context-invariant contrastive learning component to capture intrinsic factors, and a disentanglement component to extract extrinsic factors under the interplay of various contexts. The two components work together to achieve effective factor learning. Extensive experiments on real-world datasets demonstrate IEDR's effectiveness in learning disentangled factors and significantly improving recommendation accuracy by up to 4% in NDCG. 

---
# Optimizing open-domain question answering with graph-based retrieval augmented generation 

**Authors**: Joyce Cahoon, Prerna Singh, Nick Litombe, Jonathan Larson, Ha Trinh, Yiwen Zhu, Andreas Mueller, Fotis Psallidas, Carlo Curino  

**Link**: [PDF](https://arxiv.org/pdf/2503.02922)  

**Abstract**: In this work, we benchmark various graph-based retrieval-augmented generation (RAG) systems across a broad spectrum of query types, including OLTP-style (fact-based) and OLAP-style (thematic) queries, to address the complex demands of open-domain question answering (QA). Traditional RAG methods often fall short in handling nuanced, multi-document synthesis tasks. By structuring knowledge as graphs, we can facilitate the retrieval of context that captures greater semantic depth and enhances language model operations. We explore graph-based RAG methodologies and introduce TREX, a novel, cost-effective alternative that combines graph-based and vector-based retrieval techniques. Our benchmarking across four diverse datasets highlights the strengths of different RAG methodologies, demonstrates TREX's ability to handle multiple open-domain QA types, and reveals the limitations of current evaluation methods.
In a real-world technical support case study, we demonstrate how TREX solutions can surpass conventional vector-based RAG in efficiently synthesizing data from heterogeneous sources. Our findings underscore the potential of augmenting large language models with advanced retrieval and orchestration capabilities, advancing scalable, graph-based AI solutions. 

---
# A Predict-Then-Optimize Customer Allocation Framework for Online Fund Recommendation 

**Authors**: Xing Tang, Yunpeng Weng, Fuyuan Lyu, Dugang Liu, Xiuqiang He  

**Link**: [PDF](https://arxiv.org/pdf/2503.03165)  

**Abstract**: With the rapid growth of online investment platforms, funds can be distributed to individual customers online. The central issue is to match funds with potential customers under constraints. Most mainstream platforms adopt the recommendation formulation to tackle the problem. However, the traditional recommendation regime has its inherent drawbacks when applying the fund-matching problem with multiple constraints. In this paper, we model the fund matching under the allocation formulation. We design PTOFA, a Predict-Then-Optimize Fund Allocation framework. This data-driven framework consists of two stages, i.e., prediction and optimization, which aim to predict expected revenue based on customer behavior and optimize the impression allocation to achieve the maximum revenue under the necessary constraints, respectively. Extensive experiments on real-world datasets from an industrial online investment platform validate the effectiveness and efficiency of our solution. Additionally, the online A/B tests demonstrate PTOFA's effectiveness in the real-world fund recommendation scenario. 

---
# Semi-Supervised In-Context Learning: A Baseline Study 

**Authors**: Zhengyao Gu, Henry Peng Zou, Yankai Chen, Aiwei Liu, Weizhi Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03062)  

**Abstract**: Most existing work in data selection for In-Context Learning (ICL) has focused on constructing demonstrations from ground truth annotations, with limited attention given to selecting reliable self-generated annotations. In this work, we propose a three-step semi-supervised ICL framework: annotation generation, demonstration selection, and semi-supervised inference. Our baseline, Naive-SemiICL, which prompts select high-confidence self-generated demonstrations for ICL prompting, outperforms a 16-shot baseline by an average of 9.94% across 16 datasets. We further introduce IterPSD, an annotation approach that refines pseudo-demonstrations iteratively, achieving up to 6.8% additional gains in classification tasks. Lastly, we reveal a scaling law for semi-supervised ICL, where models achieve optimal performance with over 1,000 demonstrations. 

---
# Zero-Shot Multi-Label Classification of Bangla Documents: Large Decoders Vs. Classic Encoders 

**Authors**: Souvika Sarkar, Md. Najib Hasan, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2503.02993)  

**Abstract**: Bangla, a language spoken by over 300 million native speakers and ranked as the sixth most spoken language worldwide, presents unique challenges in natural language processing (NLP) due to its complex morphological characteristics and limited resources. While recent Large Decoder Based models (LLMs), such as GPT, LLaMA, and DeepSeek, have demonstrated excellent performance across many NLP tasks, their effectiveness in Bangla remains largely unexplored. In this paper, we establish the first benchmark comparing decoder-based LLMs with classic encoder-based models for Zero-Shot Multi-Label Classification (Zero-Shot-MLC) task in Bangla. Our evaluation of 32 state-of-the-art models reveals that, existing so-called powerful encoders and decoders still struggle to achieve high accuracy on the Bangla Zero-Shot-MLC task, suggesting a need for more research and resources for Bangla NLP. 

---
# ExpertGenQA: Open-ended QA generation in Specialized Domains 

**Authors**: Haz Sameen Shahgir, Chansong Lim, Jia Chen, Evangelos E. Papalexakis, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.02948)  

**Abstract**: Generating high-quality question-answer pairs for specialized technical domains remains challenging, with existing approaches facing a tradeoff between leveraging expert examples and achieving topical diversity. We present ExpertGenQA, a protocol that combines few-shot learning with structured topic and style categorization to generate comprehensive domain-specific QA pairs. Using U.S. Federal Railroad Administration documents as a test bed, we demonstrate that ExpertGenQA achieves twice the efficiency of baseline few-shot approaches while maintaining $94.4\%$ topic coverage. Through systematic evaluation, we show that current LLM-based judges and reward models exhibit strong bias toward superficial writing styles rather than content quality. Our analysis using Bloom's Taxonomy reveals that ExpertGenQA better preserves the cognitive complexity distribution of expert-written questions compared to template-based approaches. When used to train retrieval models, our generated queries improve top-1 accuracy by $13.02\%$ over baseline performance, demonstrating their effectiveness for downstream applications in technical domains. 

---
# OkraLong: A Flexible Retrieval-Augmented Framework for Long-Text Query Processing 

**Authors**: Yulong Hui, Yihao Liu, Yao Lu, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02603)  

**Abstract**: Large Language Models (LLMs) encounter challenges in efficiently processing long-text queries, as seen in applications like enterprise document analysis and financial report comprehension. While conventional solutions employ long-context processing or Retrieval-Augmented Generation (RAG), they suffer from prohibitive input expenses or incomplete information. Recent advancements adopt context compression and dynamic retrieval loops, but still sacrifice critical details or incur iterative costs. To address these limitations, we propose OkraLong, a novel framework that flexibly optimizes the entire processing workflow. Unlike prior static or coarse-grained adaptive strategies, OkraLong adopts fine-grained orchestration through three synergistic components: analyzer, organizer and executor. The analyzer characterizes the task states, which guide the organizer in dynamically scheduling the workflow. The executor carries out the execution and generates the final answer. Experimental results demonstrate that OkraLong not only enhances answer accuracy but also achieves cost-effectiveness across a variety of datasets. 

---
# MCiteBench: A Benchmark for Multimodal Citation Text Generation in MLLMs 

**Authors**: Caiyu Hu, Yikai Zhang, Tinghui Zhu, Yiwei Ye, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02589)  

**Abstract**: Multimodal Large Language Models (MLLMs) have advanced in integrating diverse modalities but frequently suffer from hallucination. A promising solution to mitigate this issue is to generate text with citations, providing a transparent chain for verification. However, existing work primarily focuses on generating citations for text-only content, overlooking the challenges and opportunities of multimodal contexts. To address this gap, we introduce MCiteBench, the first benchmark designed to evaluate and analyze the multimodal citation text generation ability of MLLMs. Our benchmark comprises data derived from academic papers and review-rebuttal interactions, featuring diverse information sources and multimodal content. We comprehensively evaluate models from multiple dimensions, including citation quality, source reliability, and answer accuracy. Through extensive experiments, we observe that MLLMs struggle with multimodal citation text generation. We also conduct deep analyses of models' performance, revealing that the bottleneck lies in attributing the correct sources rather than understanding the multimodal content. 

---
