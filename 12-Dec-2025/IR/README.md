# Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition 

**Authors**: Lingfeng Liu, Yixin Song, Dazhong Shen, Bing Yin, Hao Li, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10688)  

**Abstract**: Popularity bias fundamentally undermines the personalization capabilities of collaborative filtering (CF) models, causing them to disproportionately recommend popular items while neglecting users' genuine preferences for niche content. While existing approaches treat this as an external confounding factor, we reveal that popularity bias is an intrinsic geometric artifact of Bayesian Pairwise Ranking (BPR) optimization in CF models. Through rigorous mathematical analysis, we prove that BPR systematically organizes item embeddings along a dominant "popularity direction" where embedding magnitudes directly correlate with interaction frequency. This geometric distortion forces user embeddings to simultaneously handle two conflicting tasks-expressing genuine preference and calibrating against global popularity-trapping them in suboptimal configurations that favor popular items regardless of individual tastes. We propose Directional Decomposition and Correction (DDC), a universally applicable framework that surgically corrects this embedding geometry through asymmetric directional updates. DDC guides positive interactions along personalized preference directions while steering negative interactions away from the global popularity direction, disentangling preference from popularity at the geometric source. Extensive experiments across multiple BPR-based architectures demonstrate that DDC significantly outperforms state-of-the-art debiasing methods, reducing training loss to less than 5% of heavily-tuned baselines while achieving superior recommendation quality and fairness. Code is available in this https URL. 

---
# The Best of the Two Worlds: Harmonizing Semantic and Hash IDs for Sequential Recommendation 

**Authors**: Ziwei Liu, Yejing Wang, Qidong Liu, Zijian Zhang, Chong Chen, Wei Huang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.10388)  

**Abstract**: Conventional Sequential Recommender Systems (SRS) typically assign unique Hash IDs (HID) to construct item embeddings. These HID embeddings effectively learn collaborative information from historical user-item interactions, making them vulnerable to situations where most items are rarely consumed (the long-tail problem). Recent methods that incorporate auxiliary information often suffer from noisy collaborative sharing caused by co-occurrence signals or semantic homogeneity caused by flat dense embeddings. Semantic IDs (SIDs), with their capability of code sharing and multi-granular semantic modeling, provide a promising alternative. However, the collaborative overwhelming phenomenon hinders the further development of SID-based methods. The quantization mechanisms commonly compromise the uniqueness of identifiers required for modeling head items, creating a performance seesaw between head and tail items. To address this dilemma, we propose \textbf{\name}, a novel framework that harmonizes the SID and HID. Specifically, we devise a dual-branch modeling architecture that enables the model to capture both the multi-granular semantics within SID while preserving the unique collaborative identity of HID. Furthermore, we introduce a dual-level alignment strategy that bridges the two representations, facilitating knowledge transfer and supporting robust preference modeling. Extensive experiments on three real-world datasets show that \name~ effectively balances recommendation quality for both head and tail items while surpassing the existing baselines. The implementation code can be found online\footnote{this https URL}. 

---
# STARS: Semantic Tokens with Augmented Representations for Recommendation at Scale 

**Authors**: Han Chen, Steven Zhu, Yingrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.10149)  

**Abstract**: Real-world ecommerce recommender systems must deliver relevant items under strict tens-of-milliseconds latency constraints despite challenges such as cold-start products, rapidly shifting user intent, and dynamic context including seasonality, holidays, and promotions. We introduce STARS, a transformer-based sequential recommendation framework built for large-scale, low-latency ecommerce settings. STARS combines several innovations: dual-memory user embeddings that separate long-term preferences from short-term session intent; semantic item tokens that fuse pretrained text embeddings, learnable deltas, and LLM-derived attribute tags, strengthening content-based matching, long-tail coverage, and cold-start performance; context-aware scoring with learned calendar and event offsets; and a latency-conscious two-stage retrieval pipeline that performs offline embedding generation and online maximum inner-product search with filtering, enabling tens-of-milliseconds response times. In offline evaluations on production-scale data, STARS improves Hit@5 by more than 75 percent relative to our existing LambdaMART system. A large-scale A/B test on 6 million visits shows statistically significant lifts, including Total Orders +0.8%, Add-to-Cart on Home +2.0%, and Visits per User +0.5%. These results demonstrate that combining semantic enrichment, multi-intent modeling, and deployment-oriented design can yield state-of-the-art recommendation quality in real-world environments without sacrificing serving efficiency. 

---
# BookReconciler: An Open-Source Tool for Metadata Enrichment and Work-Level Clustering 

**Authors**: Matt Miller, Dan Sinykin, Melanie Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2512.10165)  

**Abstract**: We present BookReconciler, an open-source tool for enhancing and clustering book data. BookReconciler allows users to take spreadsheets with minimal metadata, such as book title and author, and automatically 1) add authoritative, persistent identifiers like ISBNs 2) and cluster related Expressions and Manifestations of the same Work, e.g., different translations or editions. This enhancement makes it easier to combine related collections and analyze books at scale. The tool is currently designed as an extension for OpenRefine -- a popular software application -- and connects to major bibliographic services including the Library of Congress, VIAF, OCLC, HathiTrust, Google Books, and Wikidata. Our approach prioritizes human judgment. Through an interactive interface, users can manually evaluate matches and define the contours of a Work (e.g., to include translations or not). We evaluate reconciliation performance on datasets of U.S. prize-winning books and contemporary world fiction. BookReconciler achieves near-perfect accuracy for U.S. works but lower performance for global texts, reflecting structural weaknesses in bibliographic infrastructures for non-English and global literature. Overall, BookReconciler supports the reuse of bibliographic data across domains and applications, contributing to ongoing work in digital libraries and digital humanities. 

---
# LLM-PEA: Leveraging Large Language Models Against Phishing Email Attacks 

**Authors**: Najmul Hassan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10104)  

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination. 

---
# HGC-Herd: Efficient Heterogeneous Graph Condensation via Representative Node Herding 

**Authors**: Fuyan Ou, Siqi Ai, Yulin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09947)  

**Abstract**: Heterogeneous graph neural networks (HGNNs) have demonstrated strong capability in modeling complex semantics across multi-type nodes and relations. However, their scalability to large-scale graphs remains challenging due to structural redundancy and high-dimensional node features. Existing graph condensation approaches, such as GCond, are primarily developed for homogeneous graphs and rely on gradient matching, resulting in considerable computational, memory, and optimization overhead. We propose HGC-Herd, a training-free condensation framework that generates compact yet informative heterogeneous graphs while maintaining both semantic and structural fidelity. HGC-Herd integrates lightweight feature propagation to encode multi-hop relational context and employs a class-wise herding mechanism to identify representative nodes per class, producing balanced and discriminative subsets for downstream learning tasks. Extensive experiments on ACM, DBLP, and Freebase validate that HGC-Herd attains comparable or superior accuracy to full-graph training while markedly reducing both runtime and memory consumption. These results underscore its practical value for efficient and scalable heterogeneous graph representation learning. 

---
# Benchmarking Document Parsers on Mathematical Formula Extraction from PDFs 

**Authors**: Pius Horn, Janis Keuper  

**Link**: [PDF](https://arxiv.org/pdf/2512.09874)  

**Abstract**: Correctly parsing mathematical formulas from PDFs is critical for training large language models and building scientific knowledge bases from academic literature, yet existing benchmarks either exclude formulas entirely or lack semantically-aware evaluation metrics. We introduce a novel benchmarking framework centered on synthetically generated PDFs with precise LaTeX ground truth, enabling systematic control over layout, formulas, and content characteristics. A key methodological contribution is pioneering LLM-as-a-judge for semantic formula assessment, combined with a robust two-stage matching pipeline that handles parser output inconsistencies. Through human validation on 250 formula pairs (750 ratings from 30 evaluators), we demonstrate that LLM-based evaluation achieves substantially higher correlation with human judgment (Pearson r=0.78) compared to CDM (r=0.34) and text similarity (r~0). Evaluating 20+ contemporary PDF parsers (including specialized OCR models, vision-language models, and rule-based approaches) across 100 synthetic documents with 2,000+ formulas reveals significant performance disparities. Our findings provide crucial insights for practitioners selecting parsers for downstream applications and establish a robust, scalable methodology that enables reproducible evaluation of PDF formula extraction quality. Code and benchmark data: this https URL 

---
