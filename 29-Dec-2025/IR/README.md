# Frozen LVLMs for Micro-Video Recommendation: A Systematic Study of Feature Extraction and Fusion 

**Authors**: Huatuan Sun, Yunshan Ma, Changguang Wu, Yanxin Zhang, Pengfei Wang, Xiaoyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.21863)  

**Abstract**: Frozen Large Video Language Models (LVLMs) are increasingly employed in micro-video recommendation due to their strong multimodal understanding. However, their integration lacks systematic empirical evaluation: practitioners typically deploy LVLMs as fixed black-box feature extractors without systematically comparing alternative representation strategies. To address this gap, we present the first systematic empirical study along two key design dimensions: (i) integration strategies with ID embeddings, specifically replacement versus fusion, and (ii) feature extraction paradigms, comparing LVLM-generated captions with intermediate decoder hidden states. Extensive experiments on representative LVLMs reveal three key principles: (1) intermediate hidden states consistently outperform caption-based representations, as natural-language summarization inevitably discards fine-grained visual semantics crucial for recommendation; (2) ID embeddings capture irreplaceable collaborative signals, rendering fusion strictly superior to replacement; and (3) the effectiveness of intermediate decoder features varies significantly across layers. Guided by these insights, we propose the Dual Feature Fusion (DFF) Framework, a lightweight and plug-and-play approach that adaptively fuses multi-layer representations from frozen LVLMs with item ID embeddings. DFF achieves state-of-the-art performance on two real-world micro-video recommendation benchmarks, consistently outperforming strong baselines and providing a principled approach to integrating off-the-shelf large vision-language models into micro-video recommender systems. 

---
# KG20C & KG20C-QA: Scholarly Knowledge Graph Benchmarks for Link Prediction and Question Answering 

**Authors**: Hung-Nghiep Tran, Atsuhiro Takasu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21799)  

**Abstract**: In this paper, we present KG20C and KG20C-QA, two curated datasets for advancing question answering (QA) research on scholarly data. KG20C is a high-quality scholarly knowledge graph constructed from the Microsoft Academic Graph through targeted selection of venues, quality-based filtering, and schema definition. Although KG20C has been available online in non-peer-reviewed sources such as GitHub repository, this paper provides the first formal, peer-reviewed description of the dataset, including clear documentation of its construction and specifications. KG20C-QA is built upon KG20C to support QA tasks on scholarly data. We define a set of QA templates that convert graph triples into natural language question--answer pairs, producing a benchmark that can be used both with graph-based models such as knowledge graph embeddings and with text-based models such as large language models. We benchmark standard knowledge graph embedding methods on KG20C-QA, analyze performance across relation types, and provide reproducible evaluation protocols. By officially releasing these datasets with thorough documentation, we aim to contribute a reusable, extensible resource for the research community, enabling future work in QA, reasoning, and knowledge-driven applications in the scholarly domain. The full datasets will be released at this https URL upon paper publication. 

---
# LLM-I2I: Boost Your Small Item2Item Recommendation Model with Large Language Model 

**Authors**: Yinfu Feng, Yanjing Wu, Rong Xiao, Xiaoyi Zen  

**Link**: [PDF](https://arxiv.org/pdf/2512.21595)  

**Abstract**: Item-to-Item (I2I) recommendation models are widely used in real-world systems due to their scalability, real-time capabilities, and high recommendation quality. Research to enhance I2I performance focuses on two directions: 1) model-centric approaches, which adopt deeper architectures but risk increased computational costs and deployment complexity, and 2) data-centric methods, which refine training data without altering models, offering cost-effectiveness but struggling with data sparsity and noise. To address these challenges, we propose LLM-I2I, a data-centric framework leveraging Large Language Models (LLMs) to mitigate data quality issues. LLM-I2I includes (1) an LLM-based generator that synthesizes user-item interactions for long-tail items, alleviating data sparsity, and (2) an LLM-based discriminator that filters noisy interactions from real and synthetic data. The refined data is then fused to train I2I models. Evaluated on industry (AEDS) and academic (ARD) datasets, LLM-I2I consistently improves recommendation accuracy, particularly for long-tail items. Deployed on a large-scale cross-border e-commerce platform, it boosts recall number (RN) by 6.02% and gross merchandise value (GMV) by 1.22% over existing I2I models. This work highlights the potential of LLMs in enhancing data-centric recommendation systems without modifying model architectures. 

---
# CEMG: Collaborative-Enhanced Multimodal Generative Recommendation 

**Authors**: Yuzhen Lin, Hongyi Chen, Xuanjing Chen, Shaowen Wang, Ivonne Xu, Dongming Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21543)  

**Abstract**: Generative recommendation models often struggle with two key challenges: (1) the superficial integration of collaborative signals, and (2) the decoupled fusion of multimodal features. These limitations hinder the creation of a truly holistic item representation. To overcome this, we propose CEMG, a novel Collaborative-Enhaned Multimodal Generative Recommendation framework. Our approach features a Multimodal Fusion Layer that dynamically integrates visual and textual features under the guidance of collaborative signals. Subsequently, a Unified Modality Tokenization stage employs a Residual Quantization VAE (RQ-VAE) to convert this fused representation into discrete semantic codes. Finally, in the End-to-End Generative Recommendation stage, a large language model is fine-tuned to autoregressively generate these item codes. Extensive experiments demonstrate that CEMG significantly outperforms state-of-the-art baselines. 

---
# Selective LLM-Guided Regularization for Enhancing Recommendation Models 

**Authors**: Shanglin Yang, Zhan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.21526)  

**Abstract**: Large language models provide rich semantic priors and strong reasoning capabilities, making them promising auxiliary signals for recommendation. However, prevailing approaches either deploy LLMs as standalone recommender or apply global knowledge distillation, both of which suffer from inherent drawbacks. Standalone LLM recommender are costly, biased, and unreliable across large regions of the user item space, while global distillation forces the downstream model to imitate LLM predictions even when such guidance is inaccurate. Meanwhile, recent studies show that LLMs excel particularly in re-ranking and challenging scenarios, rather than uniformly across all this http URL introduce Selective LLM Guided Regularization, a model-agnostic and computation efficient framework that activates LLM based pairwise ranking supervision only when a trainable gating mechanism informing by user history length, item popularity, and model uncertainty predicts the LLM to be reliable. All LLM scoring is performed offline, transferring knowledge without increasing inference cost. Experiments across multiple datasets show that this selective strategy consistently improves overall accuracy and yields substantial gains in cold start and long tail regimes, outperforming global distillation baselines. 

---
# AutoPP: Towards Automated Product Poster Generation and Optimization 

**Authors**: Jiahao Fan, Yuxin Qin, Wei Feng, Yanyin Chen, Yaoyu Li, Ao Ma, Yixiu Li, Li Zhuang, Haoyi Bian, Zheng Zhang, Jingjing Lv, Junjie Shen, Ching Law  

**Link**: [PDF](https://arxiv.org/pdf/2512.21921)  

**Abstract**: Product posters blend striking visuals with informative text to highlight the product and capture customer attention. However, crafting appealing posters and manually optimizing them based on online performance is laborious and resource-consuming. To address this, we introduce AutoPP, an automated pipeline for product poster generation and optimization that eliminates the need for human intervention. Specifically, the generator, relying solely on basic product information, first uses a unified design module to integrate the three key elements of a poster (background, text, and layout) into a cohesive output. Then, an element rendering module encodes these elements into condition tokens, efficiently and controllably generating the product poster. Based on the generated poster, the optimizer enhances its Click-Through Rate (CTR) by leveraging online feedback. It systematically replaces elements to gather fine-grained CTR comparisons and utilizes Isolated Direct Preference Optimization (IDPO) to attribute CTR gains to isolated elements. Our work is supported by AutoPP1M, the largest dataset specifically designed for product poster generation and optimization, which contains one million high-quality posters and feedback collected from over one million users. Experiments demonstrate that AutoPP achieves state-of-the-art results in both offline and online settings. Our code and dataset are publicly available at: this https URL 

---
# Dynamic Cooperative Strategies in Search Engine Advertising Market: With and Without Retail Competition 

**Authors**: Huiran Li, Qiucheng Li, Baozhu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.21501)  

**Abstract**: In search engine advertising (SEA) market, where competition among retailers is intense and multifaceted, channel coordination between retailers and manufacturers emerges as a critical factor, which significantly influences the effectiveness of advertising strategies. This research attempts to provide managerial guidelines for cooperative advertising in the SEA context by modeling two cooperative advertising decision scenarios. Scenario I defines a simple cooperative channel consisting of one manufacturer and one retailer. In Scenario II, we consider a more general setting where there is an independent retailer who competes with the Manufacturer-Retailer alliance in Scenario I. We propose a novel cooperative advertising optimization model, wherein a manufacturer can advertise product directly through SEA campaigns and indirectly by subsidizing its retailer. To highlight the distinctive features of SEA, our model incorporates dynamic quality scores and focuses on a finite time horizon. In each scenario, we provide a feasible equilibrium solution of optimal policies for all members. Subsequently, we conduct numerical experiments to perform sensitivity analysis for both the quality score and gross margin. Additionally, we explore the impact of the initial market share of the competing retailer in Scenario II. Finally, we investigate how retail competition affects the cooperative alliance's optimal strategy and channel performance. Our identified properties derived from the equilibrium and numerical analyses offer crucial insights for participants engaged in cooperative advertising within the SEA market. 

---
