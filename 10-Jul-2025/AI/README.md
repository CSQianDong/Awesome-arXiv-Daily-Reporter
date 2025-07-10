# First Return, Entropy-Eliciting Explore 

**Authors**: Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, Qian Liu, Ge Zhang, Zejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.07017)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) improves the reasoning abilities of Large Language Models (LLMs) but it struggles with unstable exploration. We propose FR3E (First Return, Entropy-Eliciting Explore), a structured exploration framework that identifies high-uncertainty decision points in reasoning trajectories and performs targeted rollouts to construct semantically grounded intermediate feedback. Our method provides targeted guidance without relying on dense supervision. Empirical results on mathematical reasoning benchmarks(AIME24) show that FR3E promotes more stable training, produces longer and more coherent responses, and increases the proportion of fully correct trajectories. These results highlight the framework's effectiveness in improving LLM reasoning through more robust and structured exploration. 

---
# The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation 

**Authors**: Jieren Deng, Aleksandar Cvetkovic, Pak Kiu Chung, Dragomir Yankov, Chiqun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06993)  

**Abstract**: Traditional travel-planning systems are often static and fragmented, leaving them ill-equipped to handle real-world complexities such as evolving environmental conditions and unexpected itinerary disruptions. In this paper, we identify three gaps between existing service providers causing frustrating user experience: intelligent trip planning, precision "last-100-meter" navigation, and dynamic itinerary adaptation. We propose three cooperative agents: a Travel Planning Agent that employs grid-based spatial grounding and map analysis to help resolve complex multi-modal user queries; a Destination Assistant Agent that provides fine-grained guidance for the final navigation leg of each journey; and a Local Discovery Agent that leverages image embeddings and Retrieval-Augmented Generation (RAG) to detect and respond to trip plan disruptions. With evaluations and experiments, our system demonstrates substantial improvements in query interpretation, navigation accuracy, and disruption resilience, underscoring its promise for applications from urban exploration to emergency response. 

---
# Scaling Towards the Information Boundary of Instruction Set: InfinityInstruct-Subject Technical Report 

**Authors**: Li Du, Hanyu Zhao, Yiming Ju, Tengfei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06968)  

**Abstract**: Instruction tuning has become a foundation for unlocking the capabilities of large-scale pretrained models and improving their performance on complex tasks. Thus, the construction of high-quality instruction datasets is crucial for enhancing model performance and generalizability. Although current instruction datasets have reached tens of millions of samples, models finetuned on them may still struggle with complex instruction following and tasks in rare domains. This is primarily due to limited expansion in both ``coverage'' (coverage of task types and knowledge areas) and ``depth'' (instruction complexity) of the instruction set. To address this issue, we propose a systematic instruction data construction framework, which integrates a hierarchical labeling system, an informative seed selection algorithm, an evolutionary data synthesis process, and a model deficiency diagnosis with targeted data generation. These components form an iterative closed-loop to continuously enhance the coverage and depth of instruction data. Based on this framework, we construct InfinityInstruct-Subject, a high-quality dataset containing ~1.5 million instructions. Experiments on multiple foundation models and benchmark tasks demonstrate its effectiveness in improving instruction-following capabilities. Further analyses suggest that InfinityInstruct-Subject shows enlarged coverage and depth compared to comparable synthesized instruction datasets. Our work lays a theoretical and practical foundation for the efficient, continuous evolution of instruction datasets, moving from data quantity expansion to qualitative improvement. 

---
# SCC-recursiveness in infinite argumentation (extended version) 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06852)  

**Abstract**: Argumentation frameworks (AFs) are a foundational tool in artificial intelligence for modeling structured reasoning and conflict. SCC-recursiveness is a well-known design principle in which the evaluation of arguments is decomposed according to the strongly connected components (SCCs) of the attack graph, proceeding recursively from "higher" to "lower" components. While SCC-recursive semantics such as \cft and \stgt have proven effective for finite AFs, Baumann and Spanring showed the failure of SCC-recursive semantics to generalize reliably to infinite AFs due to issues with well-foundedness.
We propose two approaches to extending SCC-recursiveness to the infinite setting. We systematically evaluate these semantics using Baroni and Giacomin's established criteria, showing in particular that directionality fails in general. We then examine these semantics' behavior in finitary frameworks, where we find some of our semantics satisfy directionality. These results advance the theory of infinite argumentation and lay the groundwork for reasoning systems capable of handling unbounded or evolving domains. 

---
# Comparing Dialectical Systems: Contradiction and Counterexample in Belief Change (Extended Version) 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06798)  

**Abstract**: Dialectical systems are a mathematical formalism for modeling an agent updating a knowledge base seeking consistency. Introduced in the 1970s by Roberto Magari, they were originally conceived to capture how a working mathematician or a research community refines beliefs in the pursuit of truth. Dialectical systems also serve as natural models for the belief change of an automated agent, offering a unifying, computable framework for dynamic belief management.
The literature distinguishes three main models of dialectical systems: (d-)dialectical systems based on revising beliefs when they are seen to be inconsistent, p-dialectical systems based on revising beliefs based on finding a counterexample, and q-dialectical systems which can do both. We answer an open problem in the literature by proving that q-dialectical systems are strictly more powerful than p-dialectical systems, which are themselves known to be strictly stronger than (d-)dialectical systems. This result highlights the complementary roles of counterexample and contradiction in automated belief revision, and thus also in the reasoning processes of mathematicians and research communities. 

---
# Jolting Technologies: Superexponential Acceleration in AI Capabilities and Implications for AGI 

**Authors**: David Orban  

**Link**: [PDF](https://arxiv.org/pdf/2507.06398)  

**Abstract**: This paper investigates the Jolting Technologies Hypothesis, which posits superexponential growth (increasing acceleration, or a positive third derivative) in the development of AI capabilities. We develop a theoretical framework and validate detection methodologies through Monte Carlo simulations, while acknowledging that empirical validation awaits suitable longitudinal data. Our analysis focuses on creating robust tools for future empirical studies and exploring the potential implications should the hypothesis prove valid. The study examines how factors such as shrinking idea-to-action intervals and compounding iterative AI improvements drive this jolting pattern. By formalizing jolt dynamics and validating detection methods through simulation, this work provides the mathematical foundation necessary for understanding potential AI trajectories and their consequences for AGI emergence, offering insights for research and policy. 

---
# Representing Prompting Patterns with PDL: Compliance Agent Case Study 

**Authors**: Mandana Vaziri, Louis Mandel, Yuji Watanabe, Hirokuni Kitahara, Martin Hirzel, Anca Sailer  

**Link**: [PDF](https://arxiv.org/pdf/2507.06396)  

**Abstract**: Prompt engineering for LLMs remains complex, with existing frameworks either hiding complexity behind restrictive APIs or providing inflexible canned patterns that resist customization -- making sophisticated agentic programming challenging. We present the Prompt Declaration Language (PDL), a novel approach to prompt representation that tackles this fundamental complexity by bringing prompts to the forefront, enabling manual and automatic prompt tuning while capturing the composition of LLM calls together with rule-based code and external tools. By abstracting away the plumbing for such compositions, PDL aims at improving programmer productivity while providing a declarative representation that is amenable to optimization. This paper demonstrates PDL's utility through a real-world case study of a compliance agent. Tuning the prompting pattern of this agent yielded up to 4x performance improvement compared to using a canned agent and prompt pattern. 

---
# Digital Wargames to Enhance Military Medical Evacuation Decision-Making 

**Authors**: Jeremy Fischer, Ram Krishnamoorthy, Vishal Kumar, Mahdi Al-Husseini  

**Link**: [PDF](https://arxiv.org/pdf/2507.06373)  

**Abstract**: Medical evacuation is one of the United States Army's most storied and critical mission sets, responsible for efficiently and expediently evacuating the battlefield ill and injured. Medical evacuation planning involves designing a robust network of medical platforms and facilities capable of moving and treating large numbers of casualties. Until now, there has not been a medium to simulate these networks in a classroom setting and evaluate both offline planning and online decision-making performance. This work describes the Medical Evacuation Wargaming Initiative (MEWI), a three-dimensional multiplayer simulation developed in Unity that replicates battlefield constraints and uncertainties. MEWI accurately models patient interactions at casualty collection points, ambulance exchange points, medical treatment facilities, and evacuation platforms. Two operational scenarios are introduced: an amphibious island assault in the Pacific and a Eurasian conflict across a sprawling road and river network. These scenarios pit students against the clock to save as many casualties as possible while adhering to doctrinal lessons learned during didactic training. We visualize performance data collected from two iterations of the MEWI Pacific scenario executed in the United States Army's Medical Evacuation Doctrine Course. We consider post-wargame Likert survey data from student participants and external observer notes to identify key planning decision points, document medical evacuation lessons learned, and quantify general utility. Results indicate that MEWI participation substantially improves uptake of medical evacuation lessons learned and co-operative decision-making. MEWI is a substantial step forward in the field of high-fidelity training tools for medical education, and our study findings offer critical insights into improving medical evacuation education and operations across the joint force. 

---
# An AI Approach for Learning the Spectrum of the Laplace-Beltrami Operator 

**Authors**: Yulin An, Enrique del Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2507.07073)  

**Abstract**: The spectrum of the Laplace-Beltrami (LB) operator is central in geometric deep learning tasks, capturing intrinsic properties of the shape of the object under consideration. The best established method for its estimation, from a triangulated mesh of the object, is based on the Finite Element Method (FEM), and computes the top k LB eigenvalues with a complexity of O(Nk), where N is the number of points. This can render the FEM method inefficient when repeatedly applied to databases of CAD mechanical parts, or in quality control applications where part metrology is acquired as large meshes and decisions about the quality of each part are needed quickly and frequently. As a solution to this problem, we present a geometric deep learning framework to predict the LB spectrum efficiently given the CAD mesh of a part, achieving significant computational savings without sacrificing accuracy, demonstrating that the LB spectrum is learnable. The proposed Graph Neural Network architecture uses a rich set of part mesh features - including Gaussian curvature, mean curvature, and principal curvatures. In addition to our trained network, we make available, for repeatability, a large curated dataset of real-world mechanical CAD models derived from the publicly available ABC dataset used for training and testing. Experimental results show that our method reduces computation time of the LB spectrum by approximately 5 times over linear FEM while delivering competitive accuracy. 

---
# Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach 

**Authors**: Adrian S. Roman, Iran R. Roman, Juan P. Bello  

**Link**: [PDF](https://arxiv.org/pdf/2507.07066)  

**Abstract**: Acoustic mapping techniques have long been used in spatial audio processing for direction of arrival estimation (DoAE). Traditional beamforming methods for acoustic mapping, while interpretable, often rely on iterative solvers that can be computationally intensive and sensitive to acoustic variability. On the other hand, recent supervised deep learning approaches offer feedforward speed and robustness but require large labeled datasets and lack interpretability. Despite their strengths, both methods struggle to consistently generalize across diverse acoustic setups and array configurations, limiting their broader applicability. We introduce the Latent Acoustic Mapping (LAM) model, a self-supervised framework that bridges the interpretability of traditional methods with the adaptability and efficiency of deep learning methods. LAM generates high-resolution acoustic maps, adapts to varying acoustic conditions, and operates efficiently across different microphone arrays. We assess its robustness on DoAE using the LOCATA and STARSS benchmarks. LAM achieves comparable or superior localization performance to existing supervised methods. Additionally, we show that LAM's acoustic maps can serve as effective features for supervised models, further enhancing DoAE accuracy and underscoring its potential to advance adaptive, high-performance sound localization systems. 

---
# DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning 

**Authors**: Shreyas Vinaya Sathyanarayana, Rahil Shah, Sharanabasava D. Hiremath, Rishikesh Panda, Rahul Jana, Riya Singh, Rida Irfan, Ashwin Murali, Bharath Ramsundar  

**Link**: [PDF](https://arxiv.org/pdf/2507.07060)  

**Abstract**: Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses. 

---
# Comparative Analysis of CNN and Transformer Architectures with Heart Cycle Normalization for Automated Phonocardiogram Classification 

**Authors**: Martin Sondermann, Pinar Bisgin, Niklas Tschorn, Anja Burmann, Christoph M. Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2507.07058)  

**Abstract**: The automated classification of phonocardiogram (PCG) recordings represents a substantial advancement in cardiovascular diagnostics. This paper presents a systematic comparison of four distinct models for heart murmur detection: two specialized convolutional neural networks (CNNs) and two zero-shot universal audio transformers (BEATs), evaluated using fixed-length and heart cycle normalization approaches. Utilizing the PhysioNet2022 dataset, a custom heart cycle normalization method tailored to individual cardiac rhythms is introduced. The findings indicate the following AUROC values: the CNN model with fixed-length windowing achieves 79.5%, the CNN model with heart cycle normalization scores 75.4%, the BEATs transformer with fixed-length windowing achieves 65.7%, and the BEATs transformer with heart cycle normalization results in 70.1%.
The findings indicate that physiological signal constraints, especially those introduced by different normalization strategies, have a substantial impact on model performance. The research provides evidence-based guidelines for architecture selection in clinical settings, emphasizing the need for a balance between accuracy and computational efficiency. Although specialized CNNs demonstrate superior performance overall, the zero-shot transformer models may offer promising efficiency advantages during development, such as faster training and evaluation cycles, despite their lower classification accuracy. These findings highlight the potential of automated classification systems to enhance cardiac diagnostics and improve patient care. 

---
# A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering 

**Authors**: Shahana Yasmin Chowdhury, Bithi Banik, Md Tamjidul Hoque, Shreya Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2507.07046)  

**Abstract**: Nowadays, speech emotion recognition (SER) plays a vital role in the field of human-computer interaction (HCI) and the evolution of artificial intelligence (AI). Our proposed DCRF-BiLSTM model is used to recognize seven emotions: neutral, happy, sad, angry, fear, disgust, and surprise, which are trained on five datasets: RAVDESS (R), TESS (T), SAVEE (S), EmoDB (E), and Crema-D (C). The model achieves high accuracy on individual datasets, including 97.83% on RAVDESS, 97.02% on SAVEE, 95.10% for CREMA-D, and a perfect 100% on both TESS and EMO-DB. For the combined (R+T+S) datasets, it achieves 98.82% accuracy, outperforming previously reported results. To our knowledge, no existing study has evaluated a single SER model across all five benchmark datasets (i.e., R+T+S+C+E) simultaneously. In our work, we introduce this comprehensive combination and achieve a remarkable overall accuracy of 93.76%. These results confirm the robustness and generalizability of our DCRF-BiLSTM framework across diverse datasets. 

---
# Advances in Intelligent Hearing Aids: Deep Learning Approaches to Selective Noise Cancellation 

**Authors**: Haris Khan, Shumaila Asif, Hassan Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2507.07043)  

**Abstract**: The integration of artificial intelligence into hearing assistance marks a paradigm shift from traditional amplification-based systems to intelligent, context-aware audio processing. This systematic literature review evaluates advances in AI-driven selective noise cancellation (SNC) for hearing aids, highlighting technological evolution, implementation challenges, and future research directions. We synthesize findings across deep learning architectures, hardware deployment strategies, clinical validation studies, and user-centric design. The review traces progress from early machine learning models to state-of-the-art deep networks, including Convolutional Recurrent Networks for real-time inference and Transformer-based architectures for high-accuracy separation. Key findings include significant gains over traditional methods, with recent models achieving up to 18.3 dB SI-SDR improvement on noisy-reverberant benchmarks, alongside sub-10 ms real-time implementations and promising clinical outcomes. Yet, challenges remain in bridging lab-grade models with real-world deployment - particularly around power constraints, environmental variability, and personalization. Identified research gaps include hardware-software co-design, standardized evaluation protocols, and regulatory considerations for AI-enhanced hearing devices. Future work must prioritize lightweight models, continual learning, contextual-based classification and clinical translation to realize transformative hearing solutions for millions globally. 

---
# Modeling Heterogeneity across Varying Spatial Extents: Discovering Linkages between Sea Ice Retreat and Ice Shelve Melt in the Antarctic 

**Authors**: Maloy Kumar Devnath, Sudip Chakraborty, Vandana P. Janeja  

**Link**: [PDF](https://arxiv.org/pdf/2507.07036)  

**Abstract**: Spatial phenomena often exhibit heterogeneity across spatial extents and in proximity, making them complex to model-especially in dynamic regions like ice shelves and sea ice. In this study, we address this challenge by exploring the linkages between sea ice retreat and Antarctic ice shelf (AIS) melt. Although atmospheric forcing and basal melting have been widely studied, the direct impact of sea ice retreat on AIS mass loss remains underexplored. Traditional models treat sea ice and AIS as separate systems. It limits their ability to capture localized linkages and cascading feedback. To overcome this, we propose Spatial-Link, a novel graph-based framework that quantifies spatial heterogeneity to capture linkages between sea ice retreat and AIS melt. Our method constructs a spatial graph using Delaunay triangulation of satellite-derived ice change matrices, where nodes represent regions of significant change and edges encode proximity and directional consistency. We extract and statistically validate linkage paths using breadth-first search and Monte Carlo simulations. Results reveal non-local, spatially heterogeneous coupling patterns, suggesting sea ice loss can initiate or amplify downstream AIS melt. Our analysis shows how sea ice retreat evolves over an oceanic grid and progresses toward ice shelves-establishing a direct linkage. To our knowledge, this is the first proposed methodology linking sea ice retreat to AIS melt. Spatial-Link offers a scalable, data-driven tool to improve sea-level rise projections and inform climate adaptation strategies. 

---
# Surrogate Model for Heat Transfer Prediction in Impinging Jet Arrays using Dynamic Inlet/Outlet and Flow Rate Control 

**Authors**: Mikael Vaillant, Victor Oliveira Ferreira, Wiebke Mainville, Jean-Michel Lamarre, Vincent Raymond, Moncef Chioua, Bruno Blais  

**Link**: [PDF](https://arxiv.org/pdf/2507.07034)  

**Abstract**: This study presents a surrogate model designed to predict the Nusselt number distribution in an enclosed impinging jet arrays, where each jet function independently and where jets can be transformed from inlets to outlets, leading to a vast number of possible flow arrangements. While computational fluid dynamics (CFD) simulations can model heat transfer with high fidelity, their cost prohibits real-time application such as model-based temperature control. To address this, we generate a CNN-based surrogate model that can predict the Nusselt distribution in real time. We train it with data from implicit large eddy computational fluid dynamics simulations (Re < 2,000). We train two distinct models, one for a five by one array of jets (83 simulations) and one for a three by three array of jets (100 simulations). We introduce a method to extrapolate predictions to higher Reynolds numbers (Re < 10,000) using a correlation-based scaling. The surrogate models achieve high accuracy, with a normalized mean average error below 2% on validation data for the five by one surrogate model and 0.6% for the three by three surrogate model. Experimental validation confirms the model's predictive capabilities. This work provides a foundation for model-based control strategies in advanced thermal management applications. 

---
# PLAME: Leveraging Pretrained Language Models to Generate Enhanced Protein Multiple Sequence Alignments 

**Authors**: Hanqun Cao, Xinyi Zhou, Zijun Gao, Chenyu Wang, Xin Gao, Zhi Zhang, Chunbin Gu, Ge Liu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2507.07032)  

**Abstract**: Protein structure prediction is essential for drug discovery and understanding biological functions. While recent advancements like AlphaFold have achieved remarkable accuracy, most folding models rely heavily on multiple sequence alignments (MSAs) to boost prediction performance. This dependency limits their effectiveness on low-homology proteins and orphan proteins, where MSA information is sparse or unavailable. To address this limitation, we propose PLAME, a novel MSA design model that leverages evolutionary embeddings from pretrained protein language models. Unlike existing methods, PLAME introduces pretrained representations to enhance evolutionary information and employs a conservation-diversity loss to enhance generation quality. Additionally, we propose a novel MSA selection method to effectively screen high-quality MSAs and improve folding performance. We also propose a sequence quality assessment metric that provides an orthogonal perspective to evaluate MSA quality. On the AlphaFold2 benchmark of low-homology and orphan proteins, PLAME achieves state-of-the-art performance in folding enhancement and sequence quality assessment, with consistent improvements demonstrated on AlphaFold3. Ablation studies validate the effectiveness of the MSA selection method, while extensive case studies on various protein types provide insights into the relationship between AlphaFold's prediction quality and MSA characteristics. Furthermore, we demonstrate that PLAME can serve as an adapter achieving AlphaFold2-level accuracy with the ESMFold's inference speed. 

---
# Design and Implementation of an OCR-Powered Pipeline for Table Extraction from Invoices 

**Authors**: Parshva Dhilankumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.07029)  

**Abstract**: This paper presents the design and development of an OCR-powered pipeline for efficient table extraction from invoices. The system leverages Tesseract OCR for text recognition and custom post-processing logic to detect, align, and extract structured tabular data from scanned invoice documents. Our approach includes dynamic preprocessing, table boundary detection, and row-column mapping, optimized for noisy and non-standard invoice formats. The resulting pipeline significantly improves data extraction accuracy and consistency, supporting real-world use cases such as automated financial workflows and digital archiving. 

---
# FlexOlmo: Open Language Models for Flexible Data Use 

**Authors**: Weijia Shi, Akshita Bhagia, Kevin Farhat, Niklas Muennighoff, Pete Walsh, Jacob Morrison, Dustin Schwenk, Shayne Longpre, Jake Poznanski, Allyson Ettinger, Daogao Liu, Margaret Li, Dirk Groeneveld, Mike Lewis, Wen-tau Yih, Luca Soldaini, Kyle Lo, Noah A. Smith, Luke Zettlemoyer, Pang Wei Koh, Hannaneh Hajishirzi, Ali Farhadi, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2507.07024)  

**Abstract**: We introduce FlexOlmo, a new class of language models (LMs) that supports (1) distributed training without data sharing, where different model parameters are independently trained on closed datasets, and (2) data-flexible inference, where these parameters along with their associated data can be flexibly included or excluded from model inferences with no further training. FlexOlmo employs a mixture-of-experts (MoE) architecture where each expert is trained independently on closed datasets and later integrated through a new domain-informed routing without any joint training. FlexOlmo is trained on FlexMix, a corpus we curate comprising publicly available datasets alongside seven domain-specific sets, representing realistic approximations of closed sets. We evaluate models with up to 37 billion parameters (20 billion active) on 31 diverse downstream tasks. We show that a general expert trained on public data can be effectively combined with independently trained experts from other data owners, leading to an average 41% relative improvement while allowing users to opt out of certain data based on data licensing or permission requirements. Our approach also outperforms prior model merging methods by 10.1% on average and surpasses the standard MoE trained without data restrictions using the same training FLOPs. Altogether, this research presents a solution for both data owners and researchers in regulated industries with sensitive or protected data. FlexOlmo enables benefiting from closed data while respecting data owners' preferences by keeping their data local and supporting fine-grained control of data access during inference. 

---
# Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing 

**Authors**: Eunbyeol Cho, Jiyoun Kim, Minjae Lee, Sungjin Park, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06996)  

**Abstract**: Electronic Health Records (EHR) are time-series relational databases that record patient interactions and medical events over time, serving as a critical resource for healthcare research and applications. However, privacy concerns and regulatory restrictions limit the sharing and utilization of such sensitive data, necessitating the generation of synthetic EHR datasets. Unlike previous EHR synthesis methods, which typically generate medical records consisting of expert-chosen features (e.g. a few vital signs or structured codes only), we introduce RawMed, the first framework to synthesize multi-table, time-series EHR data that closely resembles raw EHRs. Using text-based representation and compression techniques, RawMed captures complex structures and temporal dynamics with minimal preprocessing. We also propose a new evaluation framework for multi-table time-series synthetic EHRs, assessing distributional similarity, inter-table relationships, temporal dynamics, and privacy. Validated on two open-source EHR datasets, RawMed outperforms baseline models in fidelity and utility. The code is available at this https URL. 

---
# Cross-Modality Masked Learning for Survival Prediction in ICI Treated NSCLC Patients 

**Authors**: Qilong Xing, Zikai Song, Bingxin Gong, Lian Yang, Junqing Yu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06994)  

**Abstract**: Accurate prognosis of non-small cell lung cancer (NSCLC) patients undergoing immunotherapy is essential for personalized treatment planning, enabling informed patient decisions, and improving both treatment outcomes and quality of life. However, the lack of large, relevant datasets and effective multi-modal feature fusion strategies pose significant challenges in this domain. To address these challenges, we present a large-scale dataset and introduce a novel framework for multi-modal feature fusion aimed at enhancing the accuracy of survival prediction. The dataset comprises 3D CT images and corresponding clinical records from NSCLC patients treated with immune checkpoint inhibitors (ICI), along with progression-free survival (PFS) and overall survival (OS) data. We further propose a cross-modality masked learning approach for medical feature fusion, consisting of two distinct branches, each tailored to its respective modality: a Slice-Depth Transformer for extracting 3D features from CT images and a graph-based Transformer for learning node features and relationships among clinical variables in tabular data. The fusion process is guided by a masked modality learning strategy, wherein the model utilizes the intact modality to reconstruct missing components. This mechanism improves the integration of modality-specific features, fostering more effective inter-modality relationships and feature interactions. Our approach demonstrates superior performance in multi-modal integration for NSCLC survival prediction, surpassing existing methods and setting a new benchmark for prognostic models in this context. 

---
# MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation 

**Authors**: Qilong Xing, Zikai Song, Youjia Zhang, Na Feng, Junqing Yu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06992)  

**Abstract**: Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation. 

---
# Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy 

**Authors**: Bogdan Kulynych, Juan Felipe Gomez, Georgios Kaissis, Jamie Hayes, Borja Balle, Flavio du Pin Calmon, Jean Louis Raisaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06969)  

**Abstract**: Differentially private (DP) mechanisms are difficult to interpret and calibrate because existing methods for mapping standard privacy parameters to concrete privacy risks -- re-identification, attribute inference, and data reconstruction -- are both overly pessimistic and inconsistent. In this work, we use the hypothesis-testing interpretation of DP ($f$-DP), and determine that bounds on attack success can take the same unified form across re-identification, attribute inference, and data reconstruction risks. Our unified bounds are (1) consistent across a multitude of attack settings, and (2) tunable, enabling practitioners to evaluate risk with respect to arbitrary (including worst-case) levels of baseline risk. Empirically, our results are tighter than prior methods using $\varepsilon$-DP, Rényi DP, and concentrated DP. As a result, calibrating noise using our bounds can reduce the required noise by 20% at the same risk level, which yields, e.g., more than 15pp accuracy increase in a text classification task. Overall, this unifying perspective provides a principled framework for interpreting and calibrating the degree of protection in DP against specific levels of re-identification, attribute inference, or data reconstruction risk. 

---
# Noisy PDE Training Requires Bigger PINNs 

**Authors**: Sebastien Andre-Sloan, Anirbit Mukherjee, Matthew Colbrook  

**Link**: [PDF](https://arxiv.org/pdf/2507.06967)  

**Abstract**: Physics-Informed Neural Networks (PINNs) are increasingly used to approximate solutions of partial differential equations (PDEs), especially in high dimensions. In real-world applications, data samples are noisy, so it is important to know when a predictor can still achieve low empirical risk. However, little is known about the conditions under which a PINN can do so effectively. We prove a lower bound on the size of neural networks required for the supervised PINN empirical risk to fall below the variance of noisy supervision labels. Specifically, if a predictor achieves an empirical risk $O(\eta)$ below $\sigma^2$ (variance of supervision data), then necessarily $d_N\log d_N\gtrsim N_s \eta^2$, where $N_s$ is the number of samples and $d_N$ is the number of trainable parameters of the PINN. A similar constraint applies to the fully unsupervised PINN setting when boundary labels are sampled noisily. Consequently, increasing the number of noisy supervision labels alone does not provide a ``free lunch'' in reducing empirical risk. We also show empirically that PINNs can indeed achieve empirical risks below $\sigma^2$ under such conditions. As a case study, we investigate PINNs applied to the Hamilton--Jacobi--Bellman (HJB) PDE. Our findings lay the groundwork for quantitatively understanding the parameter requirements for training PINNs in the presence of noise. 

---
# CheXPO: Preference Optimization for Chest X-ray VLMs with Counterfactual Rationale 

**Authors**: Xiao Liang, Jiawei Hu, Di Wang, Zhi Ma, Lin Zhao, Ronghan Li, Bo Wan, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06959)  

**Abstract**: Vision-language models (VLMs) are prone to hallucinations that critically compromise reliability in medical applications. While preference optimization can mitigate these hallucinations through clinical feedback, its implementation faces challenges such as clinically irrelevant training samples, imbalanced data distributions, and prohibitive expert annotation costs. To address these challenges, we introduce CheXPO, a Chest X-ray Preference Optimization strategy that combines confidence-similarity joint mining with counterfactual rationale. Our approach begins by synthesizing a unified, fine-grained multi-task chest X-ray visual instruction dataset across different question types for supervised fine-tuning (SFT). We then identify hard examples through token-level confidence analysis of SFT failures and use similarity-based retrieval to expand hard examples for balancing preference sample distributions, while synthetic counterfactual rationales provide fine-grained clinical preferences, eliminating the need for additional expert input. Experiments show that CheXPO achieves 8.93% relative performance gain using only 5% of SFT samples, reaching state-of-the-art performance across diverse clinical tasks and providing a scalable, interpretable solution for real-world radiology applications. 

---
# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models 

**Authors**: Keyon Vafa, Peter G. Chang, Ashesh Rambachan, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06952)  

**Abstract**: Foundation models are premised on the idea that sequence prediction can uncover deeper domain understanding, much like how Kepler's predictions of planetary motion later led to the discovery of Newtonian mechanics. However, evaluating whether these models truly capture deeper structure remains a challenge. We develop a technique for evaluating foundation models that examines how they adapt to synthetic datasets generated from some postulated world model. Our technique measures whether the foundation model's inductive bias aligns with the world model, and so we refer to it as an inductive bias probe. Across multiple domains, we find that foundation models can excel at their training tasks yet fail to develop inductive biases towards the underlying world model when adapted to new tasks. We particularly find that foundation models trained on orbital trajectories consistently fail to apply Newtonian mechanics when adapted to new physics tasks. Further analysis reveals that these models behave as if they develop task-specific heuristics that fail to generalize. 

---
# Beyond Connectivity: An Open Architecture for AI-RAN Convergence in 6G 

**Authors**: Michele Polese, Niloofar Mohamadi, Salvatore D'Oro, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2507.06911)  

**Abstract**: The proliferation of data-intensive Artificial Intelligence (AI) applications at the network edge demands a fundamental shift in RAN design, from merely consuming AI for network optimization, to actively enabling distributed AI workloads. This paradigm shift presents a significant opportunity for network operators to monetize AI at the edge while leveraging existing infrastructure investments. To realize this vision, this article presents a novel converged O-RAN and AI-RAN architecture that unifies orchestration and management of both telecommunications and AI workloads on shared infrastructure. The proposed architecture extends the Open RAN principles of modularity, disaggregation, and cloud-nativeness to support heterogeneous AI deployments. We introduce two key architectural innovations: (i) the AI-RAN Orchestrator, which extends the O-RAN Service Management and Orchestration (SMO) to enable integrated resource and allocation across RAN and AI workloads; and (ii) AI-RAN sites that provide distributed edge AI platforms with real-time processing capabilities. The proposed system supports flexible deployment options, allowing AI workloads to be orchestrated with specific timing requirements (real-time or batch processing) and geographic targeting. The proposed architecture addresses the orchestration requirements for managing heterogeneous workloads at different time scales while maintaining open, standardized interfaces and multi-vendor interoperability. 

---
# MultiJustice: A Chinese Dataset for Multi-Party, Multi-Charge Legal Prediction 

**Authors**: Xiao Wang, Jiahuan Pei, Diancheng Shui, Zhiguang Han, Xin Sun, Dawei Zhu, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06909)  

**Abstract**: Legal judgment prediction offers a compelling method to aid legal practitioners and researchers. However, the research question remains relatively under-explored: Should multiple defendants and charges be treated separately in LJP? To address this, we introduce a new dataset namely multi-person multi-charge prediction (MPMCP), and seek the answer by evaluating the performance of several prevailing legal large language models (LLMs) on four practical legal judgment scenarios: (S1) single defendant with a single charge, (S2) single defendant with multiple charges, (S3) multiple defendants with a single charge, and (S4) multiple defendants with multiple charges. We evaluate the dataset across two LJP tasks, i.e., charge prediction and penalty term prediction. We have conducted extensive experiments and found that the scenario involving multiple defendants and multiple charges (S4) poses the greatest challenges, followed by S2, S3, and S1. The impact varies significantly depending on the model. For example, in S4 compared to S1, InternLM2 achieves approximately 4.5% lower F1-score and 2.8% higher LogD, while Lawformer demonstrates around 19.7% lower F1-score and 19.0% higher LogD. Our dataset and code are available at this https URL. 

---
# MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection 

**Authors**: Ziyan Liu, Chunxiao Fan, Haoran Lou, Yuexin Wu, Kaiwei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06908)  

**Abstract**: The rapid expansion of memes on social media has highlighted the urgent need for effective approaches to detect harmful content. However, traditional data-driven approaches struggle to detect new memes due to their evolving nature and the lack of up-to-date annotated data. To address this issue, we propose MIND, a multi-agent framework for zero-shot harmful meme detection that does not rely on annotated data. MIND implements three key strategies: 1) We retrieve similar memes from an unannotated reference set to provide contextual information. 2) We propose a bi-directional insight derivation mechanism to extract a comprehensive understanding of similar memes. 3) We then employ a multi-agent debate mechanism to ensure robust decision-making through reasoned arbitration. Extensive experiments on three meme datasets demonstrate that our proposed framework not only outperforms existing zero-shot approaches but also shows strong generalization across different model architectures and parameter scales, providing a scalable solution for harmful meme detection. The code is available at this https URL. 

---
# VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation 

**Authors**: Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.06899)  

**Abstract**: Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents. 

---
# SCoRE: Streamlined Corpus-based Relation Extraction using Multi-Label Contrastive Learning and Bayesian kNN 

**Authors**: Luca Mariotti, Veronica Guidetti, Federica Mandreoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.06895)  

**Abstract**: The growing demand for efficient knowledge graph (KG) enrichment leveraging external corpora has intensified interest in relation extraction (RE), particularly under low-supervision settings. To address the need for adaptable and noise-resilient RE solutions that integrate seamlessly with pre-trained large language models (PLMs), we introduce SCoRE, a modular and cost-effective sentence-level RE system. SCoRE enables easy PLM switching, requires no finetuning, and adapts smoothly to diverse corpora and KGs. By combining supervised contrastive learning with a Bayesian k-Nearest Neighbors (kNN) classifier for multi-label classification, it delivers robust performance despite the noisy annotations of distantly supervised corpora. To improve RE evaluation, we propose two novel metrics: Correlation Structure Distance (CSD), measuring the alignment between learned relational patterns and KG structures, and Precision at R (P@R), assessing utility as a recommender system. We also release Wiki20d, a benchmark dataset replicating real-world RE conditions where only KG-derived annotations are available. Experiments on five benchmarks show that SCoRE matches or surpasses state-of-the-art methods while significantly reducing energy consumption. Further analyses reveal that increasing model complexity, as seen in prior work, degrades performance, highlighting the advantages of SCoRE's minimal design. Combining efficiency, modularity, and scalability, SCoRE stands as an optimal choice for real-world RE applications. 

---
# Developing and Maintaining an Open-Source Repository of AI Evaluations: Challenges and Insights 

**Authors**: Alexandra Abbas, Celia Waggoner, Justin Olive  

**Link**: [PDF](https://arxiv.org/pdf/2507.06893)  

**Abstract**: AI evaluations have become critical tools for assessing large language model capabilities and safety. This paper presents practical insights from eight months of maintaining $inspect\_evals$, an open-source repository of 70+ community-contributed AI evaluations. We identify key challenges in implementing and maintaining AI evaluations and develop solutions including: (1) a structured cohort management framework for scaling community contributions, (2) statistical methodologies for optimal resampling and cross-model comparison with uncertainty quantification, and (3) systematic quality control processes for reproducibility. Our analysis reveals that AI evaluation requires specialized infrastructure, statistical rigor, and community coordination beyond traditional software development practices. 

---
# Squeeze the Soaked Sponge: Efficient Off-policy Reinforcement Finetuning for Large Language Model 

**Authors**: Jing Liang, Hongyao Tang, Yi Ma, Jinyi Liu, Yan Zheng, Shuyue Hu, Lei Bai, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06892)  

**Abstract**: Reinforcement Learning (RL) has demonstrated its potential to improve the reasoning ability of Large Language Models (LLMs). One major limitation of most existing Reinforcement Finetuning (RFT) methods is that they are on-policy RL in nature, i.e., data generated during the past learning process is not fully utilized. This inevitably comes at a significant cost of compute and time, posing a stringent bottleneck on continuing economic and efficient scaling. To this end, we launch the renaissance of off-policy RL and propose Reincarnating Mix-policy Proximal Policy Gradient (ReMix), a general approach to enable on-policy RFT methods like PPO and GRPO to leverage off-policy data. ReMix consists of three major components: (1) Mix-policy proximal policy gradient with an increased Update-To-Data (UTD) ratio for efficient training; (2) KL-Convex policy constraint to balance the trade-off between stability and flexibility; (3) Policy reincarnation to achieve a seamless transition from efficient early-stage learning to steady asymptotic improvement. In our experiments, we train a series of ReMix models upon PPO, GRPO and 1.5B, 7B base models. ReMix shows an average Pass@1 accuracy of 52.10% (for 1.5B model) with 0.079M response rollouts, 350 training steps and achieves 63.27%/64.39% (for 7B model) with 0.007M/0.011M response rollouts, 50/75 training steps, on five math reasoning benchmarks (i.e., AIME'24, AMC'23, Minerva, OlympiadBench, and MATH500). Compared with 15 recent advanced models, ReMix shows SOTA-level performance with an over 30x to 450x reduction in training cost in terms of rollout data volume. In addition, we reveal insightful findings via multifaceted analysis, including the implicit preference for shorter responses due to the Whipping Effect of off-policy discrepancy, the collapse mode of self-reflection behavior under the presence of severe off-policyness, etc. 

---
# A Single-Point Measurement Framework for Robust Cyber-Attack Diagnosis in Smart Microgrids Using Dual Fractional-Order Feature Analysis 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06890)  

**Abstract**: Cyber-attacks jeopardize the safe operation of smart microgrids. At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modelling assumptions that are untenable under single-sensor constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves low-latency fault localisation and cyber-attack detection using only one VPQ (Voltage-Power-Reactive-power) sensor. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Grünwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritise the most challenging samples. Experiments on a four-inverter microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6 % (bias), 94.0 % (noise), 92.8 % (data replacement), and 95.7 % (replay), while sustaining 96.7 % under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of smart microgrids. 

---
# Winning and losing with Artificial Intelligence: What public discourse about ChatGPT tells us about how societies make sense of technological change 

**Authors**: Adrian Rauchfleisch, Joshua Philip Suarez, Nikka Marie Sales, Andreas Jungherr  

**Link**: [PDF](https://arxiv.org/pdf/2507.06876)  

**Abstract**: Public product launches in Artificial Intelligence can serve as focusing events for collective attention, surfacing how societies react to technological change. Social media provide a window into the sensemaking around these events, surfacing hopes and fears and showing who chooses to engage in the discourse and when. We demonstrate that public sensemaking about AI is shaped by economic interests and cultural values of those involved. We analyze 3.8 million tweets posted by 1.6 million users across 117 countries in response to the public launch of ChatGPT in 2022. Our analysis shows how economic self-interest, proxied by occupational skill types in writing, programming, and mathematics, and national cultural orientations, as measured by Hofstede's individualism, uncertainty avoidance, and power distance dimensions, shape who speaks, when they speak, and their stance towards ChatGPT. Roles requiring more technical skills, such as programming and mathematics, tend to engage earlier and express more positive stances, whereas writing-centric occupations join later with greater skepticism. At the cultural level, individualism predicts both earlier engagement and a more negative stance, and uncertainty avoidance reduces the prevalence of positive stances but does not delay when users first engage with ChatGPT. Aggregate sentiment trends mask the dynamics observed in our study. The shift toward a more critical stance towards ChatGPT over time stems primarily from the entry of more skeptical voices rather than a change of heart among early adopters. Our findings underscore the importance of both the occupational background and cultural context in understanding public reactions to AI. 

---
# IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization 

**Authors**: Subrat Kishore Dutta, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06856)  

**Abstract**: Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective. 

---
# DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models 

**Authors**: Liang Wang, Yu Rong, Tingyang Xu, Zhenyi Zhong, Zhiyuan Liu, Pengju Wang, Deli Zhao, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06853)  

**Abstract**: Molecular structure elucidation from spectra is a foundational problem in chemistry, with profound implications for compound identification, synthesis, and drug development. Traditional methods rely heavily on expert interpretation and lack scalability. Pioneering machine learning methods have introduced retrieval-based strategies, but their reliance on finite libraries limits generalization to novel molecules. Generative models offer a promising alternative, yet most adopt autoregressive SMILES-based architectures that overlook 3D geometry and struggle to integrate diverse spectral modalities. In this work, we present DiffSpectra, a generative framework that directly infers both 2D and 3D molecular structures from multi-modal spectral data using diffusion models. DiffSpectra formulates structure elucidation as a conditional generation process. Its denoising network is parameterized by Diffusion Molecule Transformer, an SE(3)-equivariant architecture that integrates topological and geometric information. Conditioning is provided by SpecFormer, a transformer-based spectral encoder that captures intra- and inter-spectral dependencies from multi-modal spectra. Extensive experiments demonstrate that DiffSpectra achieves high accuracy in structure elucidation, recovering exact structures with 16.01% top-1 accuracy and 96.86% top-20 accuracy through sampling. The model benefits significantly from 3D geometric modeling, SpecFormer pre-training, and multi-modal conditioning. These results highlight the effectiveness of spectrum-conditioned diffusion modeling in addressing the challenge of molecular structure elucidation. To our knowledge, DiffSpectra is the first framework to unify multi-modal spectral reasoning and joint 2D/3D generative modeling for de novo molecular structure elucidation. 

---
# The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover 

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.06850)  

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors. 

---
# OpenDPDv2: A Unified Learning and Optimization Framework for Neural Network Digital Predistortion 

**Authors**: Yizhuo Wu, Ang Li, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06849)  

**Abstract**: Neural network (NN)-based Digital Predistortion (DPD) stands out in improving signal quality in wideband radio frequency (RF) power amplifiers (PAs) employing complex modulation. However, NN DPDs usually rely on a large number of parameters for effective linearization and can significantly contribute to the energy consumption of the digital back-end in RF systems. This paper presents OpenDPDv2, a unified framework for PA modeling, DPD learning, and model optimization to reduce power consumption while maintaining high linearization performance. The optimization techniques feature a novel DPD algorithm, TRes-DeltaGRU, alongside two energy-efficient methods. The top-performing 32-bit floating-point (FP32) TRes-DeltaGRU-DPD model achieves an Adjacent Channel Power Ratio (ACPR) of -59.4 dBc and Error Vector Magnitude (EVM) of -42.1 dBc. By exploiting fixed-point quantization and dynamic temporal sparsity of input signals and hidden neurons, the inference energy of our model can be reduced by 4.5X while still maintaining -50.3 dBc ACPR and -35.2 dB EVM with 56% temporal sparsity. This was evaluated using a TM3.1a 200 MHz bandwidth 256-QAM OFDM signal applied to a 3.5 GHz GaN Doherty RF PA. OpenDPDv2 code, datasets, and documentation are publicly accessible at: this https URL. 

---
# Physics-Grounded Motion Forecasting via Equation Discovery for Trajectory-Guided Image-to-Video Generation 

**Authors**: Tao Feng, Xianbing Zhao, Zhenhua Chen, Tien Tsin Wong, Hamid Rezatofighi, Gholamreza Haffari, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06830)  

**Abstract**: Recent advances in diffusion-based and autoregressive video generation models have achieved remarkable visual realism. However, these models typically lack accurate physical alignment, failing to replicate real-world dynamics in object motion. This limitation arises primarily from their reliance on learned statistical correlations rather than capturing mechanisms adhering to physical laws. To address this issue, we introduce a novel framework that integrates symbolic regression (SR) and trajectory-guided image-to-video (I2V) models for physics-grounded video forecasting. Our approach extracts motion trajectories from input videos, uses a retrieval-based pre-training mechanism to enhance symbolic regression, and discovers equations of motion to forecast physically accurate future trajectories. These trajectories then guide video generation without requiring fine-tuning of existing models. Evaluated on scenarios in Classical Mechanics, including spring-mass, pendulums, and projectile motions, our method successfully recovers ground-truth analytical equations and improves the physical alignment of generated videos over baseline methods. 

---
# Speckle2Self: Self-Supervised Ultrasound Speckle Reduction Without Clean Data 

**Authors**: Xuesong Li, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06828)  

**Abstract**: Image denoising is a fundamental task in computer vision, particularly in medical ultrasound (US) imaging, where speckle noise significantly degrades image quality. Although recent advancements in deep neural networks have led to substantial improvements in denoising for natural images, these methods cannot be directly applied to US speckle noise, as it is not purely random. Instead, US speckle arises from complex wave interference within the body microstructure, making it tissue-dependent. This dependency means that obtaining two independent noisy observations of the same scene, as required by pioneering Noise2Noise, is not feasible. Additionally, blind-spot networks also cannot handle US speckle noise due to its high spatial dependency. To address this challenge, we introduce Speckle2Self, a novel self-supervised algorithm for speckle reduction using only single noisy observations. The key insight is that applying a multi-scale perturbation (MSP) operation introduces tissue-dependent variations in the speckle pattern across different scales, while preserving the shared anatomical structure. This enables effective speckle suppression by modeling the clean image as a low-rank signal and isolating the sparse noise component. To demonstrate its effectiveness, Speckle2Self is comprehensively compared with conventional filter-based denoising algorithms and SOTA learning-based methods, using both realistic simulated US images and human carotid US images. Additionally, data from multiple US machines are employed to evaluate model generalization and adaptability to images from unseen domains. \textit{Code and datasets will be released upon acceptance. 

---
# Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning 

**Authors**: Matej Straka, Martin Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2507.06825)  

**Abstract**: We introduce a real-time strategy game environment built on this http URL, a game that hosts thousands of active players each week across multiple game formats. Our environment is fully compatible with Gymnasium and PettingZoo, capable of running thousands of frames per second on commodity hardware. Our reference agent -- trained with supervised pre-training and self-play -- hits the top 0.003\% of the 1v1 human leaderboard after just 36 hours on a single H100 GPU. To accelerate learning, we incorporate potential-based reward shaping and memory features. Our contributions -- a modular RTS benchmark and a competitive, state-of-the-art baseline agent -- provide an accessible yet challenging platform for advancing multi-agent reinforcement learning research. 

---
# HeLo: Heterogeneous Multi-Modal Fusion with Label Correlation for Emotion Distribution Learning 

**Authors**: Chuhang Zheng, Chunwei Tian, Jie Wen, Daoqiang Zhang, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06821)  

**Abstract**: Multi-modal emotion recognition has garnered increasing attention as it plays a significant role in human-computer interaction (HCI) in recent years. Since different discrete emotions may exist at the same time, compared with single-class emotion recognition, emotion distribution learning (EDL) that identifies a mixture of basic emotions has gradually emerged as a trend. However, existing EDL methods face challenges in mining the heterogeneity among multiple modalities. Besides, rich semantic correlations across arbitrary basic emotions are not fully exploited. In this paper, we propose a multi-modal emotion distribution learning framework, named HeLo, aimed at fully exploring the heterogeneity and complementary information in multi-modal emotional data and label correlation within mixed basic emotions. Specifically, we first adopt cross-attention to effectively fuse the physiological data. Then, an optimal transport (OT)-based heterogeneity mining module is devised to mine the interaction and heterogeneity between the physiological and behavioral representations. To facilitate label correlation learning, we introduce a learnable label embedding optimized by correlation matrix alignment. Finally, the learnable label embeddings and label correlation matrices are integrated with the multi-modal representations through a novel label correlation-driven cross-attention mechanism for accurate emotion distribution learning. Experimental results on two publicly available datasets demonstrate the superiority of our proposed method in emotion distribution learning. 

---
# Comprehensive Evaluation of Prototype Neural Networks 

**Authors**: Philipp Schlinge, Steffen Meinert, Martin Atzmueller  

**Link**: [PDF](https://arxiv.org/pdf/2507.06819)  

**Abstract**: Prototype models are an important method for explainable artificial intelligence (XAI) and interpretable machine learning. In this paper, we perform an in-depth analysis of a set of prominent prototype models including ProtoPNet, ProtoPool and PIPNet. For their assessment, we apply a comprehensive set of metrics. In addition to applying standard metrics from literature, we propose several new metrics to further complement the analysis of model interpretability. In our experimentation, we apply the set of prototype models on a diverse set of datasets including fine-grained classification, Non-IID settings and multi-label classification to further contrast the performance. Furthermore, we also provide our code as an open-source library, which facilitates simple application of the metrics itself, as well as extensibility - providing the option for easily adding new metrics and models. this https URL 

---
# Intrinsic Training Signals for Federated Learning Aggregation 

**Authors**: Cosimo Fiorini, Matteo Mosconi, Pietro Buzzega, Riccardo Salami, Simone Calderara  

**Link**: [PDF](https://arxiv.org/pdf/2507.06813)  

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy. While existing approaches for aggregating client-specific classification heads and adapted backbone parameters require architectural modifications or loss function changes, our method uniquely leverages intrinsic training signals already available during standard optimization. We present LIVAR (Layer Importance and VARiance-based merging), which introduces: i) a variance-weighted classifier aggregation scheme using naturally emergent feature statistics, and ii) an explainability-driven LoRA merging technique based on SHAP analysis of existing update parameter patterns. Without any architectural overhead, LIVAR achieves state-of-the-art performance on multiple benchmarks while maintaining seamless integration with existing FL methods. This work demonstrates that effective model merging can be achieved solely through existing training signals, establishing a new paradigm for efficient federated model aggregation. The code will be made publicly available upon acceptance. 

---
# Democratizing High-Fidelity Co-Speech Gesture Video Generation 

**Authors**: Xu Yang, Shaoli Huang, Shenbo Xie, Xuelin Chen, Yifei Liu, Changxing Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.06812)  

**Abstract**: Co-speech gesture video generation aims to synthesize realistic, audio-aligned videos of speakers, complete with synchronized facial expressions and body gestures. This task presents challenges due to the significant one-to-many mapping between audio and visual content, further complicated by the scarcity of large-scale public datasets and high computational demands. We propose a lightweight framework that utilizes 2D full-body skeletons as an efficient auxiliary condition to bridge audio signals with visual outputs. Our approach introduces a diffusion model conditioned on fine-grained audio segments and a skeleton extracted from the speaker's reference image, predicting skeletal motions through skeleton-audio feature fusion to ensure strict audio coordination and body shape consistency. The generated skeletons are then fed into an off-the-shelf human video generation model with the speaker's reference image to synthesize high-fidelity videos. To democratize research, we present CSG-405-the first public dataset with 405 hours of high-resolution videos across 71 speech types, annotated with 2D skeletons and diverse speaker demographics. Experiments show that our method exceeds state-of-the-art approaches in visual quality and synchronization while generalizing across speakers and contexts. 

---
# Towards Solving More Challenging IMO Problems via Decoupled Reasoning and Proving 

**Authors**: Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06804)  

**Abstract**: Automated Theorem Proving (ATP) in formal languages is a foundational challenge for AI. While Large Language Models (LLMs) have driven remarkable progress, a significant gap remains between their powerful informal reasoning capabilities and their weak formal proving performance. Recent studies show that the informal accuracy exceeds 80% while formal success remains below 8% on benchmarks like PutnamBench. We argue this gap persists because current state-of-the-art provers, by tightly coupling reasoning and proving, are trained with paradigms that inadvertently punish deep reasoning in favor of shallow, tactic-based strategies. To bridge this fundamental gap, we propose a novel framework that decouples high-level reasoning from low-level proof generation. Our approach utilizes two distinct, specialized models: a powerful, general-purpose Reasoner to generate diverse, strategic subgoal lemmas, and an efficient Prover to rigorously verify them. This modular design liberates the model's full reasoning potential and bypasses the pitfalls of end-to-end training. We evaluate our method on a challenging set of post-2000 IMO problems, a problem set on which no prior open-source prover has reported success. Our decoupled framework successfully solves 5 of these problems, demonstrating a significant step towards automated reasoning on exceptionally difficult mathematical challenges. To foster future research, we release our full dataset of generated and verified lemmas for a wide range of IMO problems, available at this https URL . 

---
# Text to model via SysML: Automated generation of dynamical system computational models from unstructured natural language text via enhanced System Modeling Language diagrams 

**Authors**: Matthew Anderson Hendricks, Alice Cicirello  

**Link**: [PDF](https://arxiv.org/pdf/2507.06803)  

**Abstract**: This paper contributes to speeding up the design and deployment of engineering dynamical systems by proposing a strategy for exploiting domain and expert knowledge for the automated generation of dynamical system computational model starting from a corpus of document relevant to the dynamical system of interest and an input document describing the specific system. This strategy is implemented in five steps and, crucially, it uses system modeling language diagrams (SysML) to extract accurate information about the dependencies, attributes, and operations of components. Natural Language Processing (NLP) strategies and Large Language Models (LLMs) are employed in specific tasks to improve intermediate outputs of the SySML diagrams automated generation, such as: list of key nouns; list of extracted relationships; list of key phrases and key relationships; block attribute values; block relationships; and BDD diagram generation. The applicability of automated SysML diagram generation is illustrated with different case studies. The computational models of complex dynamical systems from SysML diagrams are then obtained via code generation and computational model generation steps. In the code generation step, NLP strategies are used for summarization, while LLMs are used for validation only. The proposed approach is not limited to a specific system, domain, or computational software. The applicability of the proposed approach is shown via an end-to-end example from text to model of a simple pendulum, showing improved performance compared to results yielded by LLMs only. 

---
# Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining: Method, Evaluation and Applications 

**Authors**: Seonwu Kim, Yohan Na, Kihun Kim, Hanhee Cho, Geun Lim, Mintae Kim, Seongik Park, Ki Hyun Kim, Youngsub Han, Byoung-Ki Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2507.06795)  

**Abstract**: The emergence of open-source large language models (LLMs) has expanded opportunities for enterprise applications; however, many organizations still lack the infrastructure to deploy and maintain large-scale models. As a result, small LLMs (sLLMs) have become a practical alternative, despite their inherent performance limitations. While Domain Adaptive Continual Pretraining (DACP) has been previously explored as a method for domain adaptation, its utility in commercial applications remains under-examined. In this study, we validate the effectiveness of applying a DACP-based recipe across diverse foundation models and service domains. Through extensive experiments and real-world evaluations, we demonstrate that DACP-applied sLLMs achieve substantial gains in target domain performance while preserving general capabilities, offering a cost-efficient and scalable solution for enterprise-level deployment. 

---
# Temporal Information Retrieval via Time-Specifier Model Merging 

**Authors**: SeungYoon Han, Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, Huije Lee, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.06782)  

**Abstract**: The rapid expansion of digital information and knowledge across structured and unstructured sources has heightened the importance of Information Retrieval (IR). While dense retrieval methods have substantially improved semantic matching for general queries, they consistently underperform on queries with explicit temporal constraints--often those containing numerical expressions and time specifiers such as ``in 2015.'' Existing approaches to Temporal Information Retrieval (TIR) improve temporal reasoning but often suffer from catastrophic forgetting, leading to reduced performance on non-temporal queries. To address this, we propose Time-Specifier Model Merging (TSM), a novel method that enhances temporal retrieval while preserving accuracy on non-temporal queries. TSM trains specialized retrievers for individual time specifiers and merges them in to a unified model, enabling precise handling of temporal constraints without compromising non-temporal retrieval. Extensive experiments on both temporal and non-temporal datasets demonstrate that TSM significantly improves performance on temporally constrained queries while maintaining strong results on non-temporal queries, consistently outperforming other baseline methods. Our code is available at this https URL . 

---
# FOLC-Net: A Federated-Optimized Lightweight Architecture for Enhanced MRI Disease Diagnosis across Axial, Coronal, and Sagittal Views 

**Authors**: Saif Ur Rehman Khan, Muhammad Nabeel Asim, Sebastian Vollmer, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2507.06763)  

**Abstract**: The framework is designed to improve performance in the analysis of combined as well as single anatomical perspectives for MRI disease diagnosis. It specifically addresses the performance degradation observed in state-of-the-art (SOTA) models, particularly when processing axial, coronal, and sagittal anatomical planes. The paper introduces the FOLC-Net framework, which incorporates a novel federated-optimized lightweight architecture with approximately 1.217 million parameters and a storage requirement of only 0.9 MB. FOLC-Net integrates Manta-ray foraging optimization (MRFO) mechanisms for efficient model structure generation, global model cloning for scalable training, and ConvNeXt for enhanced client adaptability. The model was evaluated on combined multi-view data as well as individual views, such as axial, coronal, and sagittal, to assess its robustness in various medical imaging scenarios. Moreover, FOLC-Net tests a ShallowFed model on different data to evaluate its ability to generalize beyond the training dataset. The results show that FOLC-Net outperforms existing models, particularly in the challenging sagittal view. For instance, FOLC-Net achieved an accuracy of 92.44% on the sagittal view, significantly higher than the 88.37% accuracy of study method (DL + Residual Learning) and 88.95% of DL models. Additionally, FOLC-Net demonstrated improved accuracy across all individual views, providing a more reliable and robust solution for medical image analysis in decentralized environments. FOLC-Net addresses the limitations of existing SOTA models by providing a framework that ensures better adaptability to individual views while maintaining strong performance in multi-view settings. The incorporation of MRFO, global model cloning, and ConvNeXt ensures that FOLC-Net performs better in real-world medical applications. 

---
# KAConvText: Novel Approach to Burmese Sentence Classification using Kolmogorov-Arnold Convolution 

**Authors**: Ye Kyaw Thu, Thura Aung, Thazin Myint Oo, Thepchai Supnithi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06753)  

**Abstract**: This paper presents the first application of Kolmogorov-Arnold Convolution for Text (KAConvText) in sentence classification, addressing three tasks: imbalanced binary hate speech detection, balanced multiclass news classification, and imbalanced multiclass ethnic language identification. We investigate various embedding configurations, comparing random to fastText embeddings in both static and fine-tuned settings, with embedding dimensions of 100 and 300 using CBOW and Skip-gram models. Baselines include standard CNNs and CNNs augmented with a Kolmogorov-Arnold Network (CNN-KAN). In addition, we investigated KAConvText with different classification heads - MLP and KAN, where using KAN head supports enhanced interpretability. Results show that KAConvText-MLP with fine-tuned fastText embeddings achieves the best performance of 91.23% accuracy (F1-score = 0.9109) for hate speech detection, 92.66% accuracy (F1-score = 0.9267) for news classification, and 99.82% accuracy (F1-score = 0.9982) for language identification. 

---
# DIFFUMA: High-Fidelity Spatio-Temporal Video Prediction via Dual-Path Mamba and Diffusion Enhancement 

**Authors**: Xinyu Xie, Weifeng Cao, Jun Shi, Yangyang Hu, Hui Liang, Wanyong Liang, Xiaoliang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2507.06738)  

**Abstract**: Spatio-temporal video prediction plays a pivotal role in critical domains, ranging from weather forecasting to industrial automation. However, in high-precision industrial scenarios such as semiconductor manufacturing, the absence of specialized benchmark datasets severely hampers research on modeling and predicting complex processes. To address this challenge, we make a twofold this http URL, we construct and release the Chip Dicing Lane Dataset (CHDL), the first public temporal image dataset dedicated to the semiconductor wafer dicing process. Captured via an industrial-grade vision system, CHDL provides a much-needed and challenging benchmark for high-fidelity process modeling, defect detection, and digital twin this http URL, we propose DIFFUMA, an innovative dual-path prediction architecture specifically designed for such fine-grained dynamics. The model captures global long-range temporal context through a parallel Mamba module, while simultaneously leveraging a diffusion module, guided by temporal features, to restore and enhance fine-grained spatial details, effectively combating feature degradation. Experiments demonstrate that on our CHDL benchmark, DIFFUMA significantly outperforms existing methods, reducing the Mean Squared Error (MSE) by 39% and improving the Structural Similarity (SSIM) from 0.926 to a near-perfect 0.988. This superior performance also generalizes to natural phenomena datasets. Our work not only delivers a new state-of-the-art (SOTA) model but, more importantly, provides the community with an invaluable data resource to drive future research in industrial AI. 

---
# Civil Society in the Loop: Feedback-Driven Adaptation of (L)LM-Assisted Classification in an Open-Source Telegram Monitoring Tool 

**Authors**: Milena Pustet, Elisabeth Steffen, Helena Mihaljević, Grischa Stanjek, Yannis Illies  

**Link**: [PDF](https://arxiv.org/pdf/2507.06734)  

**Abstract**: The role of civil society organizations (CSOs) in monitoring harmful online content is increasingly crucial, especially as platform providers reduce their investment in content moderation. AI tools can assist in detecting and monitoring harmful content at scale. However, few open-source tools offer seamless integration of AI models and social media monitoring infrastructures. Given their thematic expertise and contextual understanding of harmful content, CSOs should be active partners in co-developing technological tools, providing feedback, helping to improve models, and ensuring alignment with stakeholder needs and values, rather than as passive 'consumers'. However, collaborations between the open source community, academia, and civil society remain rare, and research on harmful content seldom translates into practical tools usable by civil society actors. This work in progress explores how CSOs can be meaningfully involved in an AI-assisted open-source monitoring tool of anti-democratic movements on Telegram, which we are currently developing in collaboration with CSO stakeholders. 

---
# CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.06715)  

**Abstract**: Large language models (LLMs), including zero-shot and few-shot paradigms, have shown promising capabilities in clinical text generation. However, real-world applications face two key challenges: (1) patient data is highly unstructured, heterogeneous, and scattered across multiple note types and (2) clinical notes are often long and semantically dense, making naive prompting infeasible due to context length constraints and the risk of omitting clinically relevant information.
We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a domain-specific framework for structured and clinically grounded text generation using LLMs. It incorporates a novel hierarchical chunking strategy that respects clinical document structure and introduces a task-specific dual-stage retrieval mechanism. The global stage identifies relevant note types using evidence-based queries, while the local stage extracts high-value content within those notes creating relevance at both document and section levels.
We apply the system to generate structured progress notes for individual hospital visits using 15 clinical note types from the MIMIC-III dataset. Experiments show that it preserves temporal and semantic alignment across visits, achieving an average alignment score of 87.7%, surpassing the 80.7% baseline from real clinician-authored notes. The generated outputs also demonstrate high consistency across LLMs, reinforcing deterministic behavior essential for reproducibility, reliability, and clinical trust. 

---
# Photometric Stereo using Gaussian Splatting and inverse rendering 

**Authors**: Matéo Ducastel, David Tschumperlé, Yvain Quéau  

**Link**: [PDF](https://arxiv.org/pdf/2507.06684)  

**Abstract**: Recent state-of-the-art algorithms in photometric stereo rely on neural networks and operate either through prior learning or inverse rendering optimization. Here, we revisit the problem of calibrated photometric stereo by leveraging recent advances in 3D inverse rendering using the Gaussian Splatting formalism. This allows us to parameterize the 3D scene to be reconstructed and optimize it in a more interpretable manner. Our approach incorporates a simplified model for light representation and demonstrates the potential of the Gaussian Splatting rendering engine for the photometric stereo problem. 

---
# Exploring State-Space-Model based Language Model in Music Generation 

**Authors**: Wei-Jaw Lee, Fang-Chih Hsieh, Xuanjun Chen, Fang-Duo Tsai, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06674)  

**Abstract**: The recent surge in State Space Models (SSMs), particularly the emergence of Mamba, has established them as strong alternatives or complementary modules to Transformers across diverse domains. In this work, we aim to explore the potential of Mamba-based architectures for text-to-music generation. We adopt discrete tokens of Residual Vector Quantization (RVQ) as the modeling representation and empirically find that a single-layer codebook can capture semantic information in music. Motivated by this observation, we focus on modeling a single-codebook representation and adapt SiMBA, originally designed as a Mamba-based encoder, to function as a decoder for sequence modeling. We compare its performance against a standard Transformer-based decoder. Our results suggest that, under limited-resource settings, SiMBA achieves much faster convergence and generates outputs closer to the ground truth. This demonstrates the promise of SSMs for efficient and expressive text-to-music generation. We put audio examples on Github. 

---
# Elite Polarization in European Parliamentary Speeches: a Novel Measurement Approach Using Large Language Models 

**Authors**: Gennadii Iakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2507.06658)  

**Abstract**: This project introduces a new measure of elite polarization via actor and subject detection using artificial intelligence. I identify when politicians mention one another in parliamentary speeches, note who is speaking and who is being addressed, and assess the emotional temperature behind these evaluations. This maps how elites evaluate their various out-parties, allowing us to create an index of mutual out-party hostility, that is, elite polarization. While I analyzed polarization data over the past four decades for the UK, and two decades for Hungary and Italy, my approach lays the groundwork for a twenty-year, EU-wide time-series dataset on elite polarization. I obtain the results that can be aggregated by party and quarter. The resulting index demonstrates a good face validity: it reacts to events such as electoral campaigns, country- and party-level crises, and to parties losing and assuming power. 

---
# MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text to Image Retrieval 

**Authors**: Naoya Sogi, Takashi Shibata, Makoto Terao, Masanori Suganuma, Takayuki Okatani  

**Link**: [PDF](https://arxiv.org/pdf/2507.06654)  

**Abstract**: Result diversification (RD) is a crucial technique in Text-to-Image Retrieval for enhancing the efficiency of a practical application. Conventional methods focus solely on increasing the diversity metric of image appearances. However, the diversity metric and its desired value vary depending on the application, which limits the applications of RD. This paper proposes a novel task called CDR-CA (Contextual Diversity Refinement of Composite Attributes). CDR-CA aims to refine the diversities of multiple attributes, according to the application's context. To address this task, we propose Multi-Source DPPs, a simple yet strong baseline that extends the Determinantal Point Process (DPP) to multi-sources. We model MS-DPP as a single DPP model with a unified similarity matrix based on a manifold representation. We also introduce Tangent Normalization to reflect contexts. Extensive experiments demonstrate the effectiveness of the proposed method. Our code is publicly available at this https URL. 

---
# Deep Disentangled Representation Network for Treatment Effect Estimation 

**Authors**: Hui Meng, Keping Yang, Xuyu Peng, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06650)  

**Abstract**: Estimating individual-level treatment effect from observational data is a fundamental problem in causal inference and has attracted increasing attention in the fields of education, healthcare, and public this http URL this work, we concentrate on the study of disentangled representation methods that have shown promising outcomes by decomposing observed covariates into instrumental, confounding, and adjustment factors. However, most of the previous work has primarily revolved around generative models or hard decomposition methods for covariates, which often struggle to guarantee the attainment of precisely disentangled factors. In order to effectively model different causal relationships, we propose a novel treatment effect estimation algorithm that incorporates a mixture of experts with multi-head attention and a linear orthogonal regularizer to softly decompose the pre-treatment variables, and simultaneously eliminates selection bias via importance sampling re-weighting techniques. We conduct extensive experiments on both public semi-synthetic and real-world production datasets. The experimental results clearly demonstrate that our algorithm outperforms the state-of-the-art methods focused on individual treatment effects. 

---
# EXAONE Path 2.0: Pathology Foundation Model with End-to-End Supervision 

**Authors**: Myungjang Pyeon, Janghyeon Lee, Minsoo Lee, Juseung Yun, Hwanil Choi, Jonghyun Kim, Jiwon Kim, Yi Hu, Jongseong Jang, Soonyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.06639)  

**Abstract**: In digital pathology, whole-slide images (WSIs) are often difficult to handle due to their gigapixel scale, so most approaches train patch encoders via self-supervised learning (SSL) and then aggregate the patch-level embeddings via multiple instance learning (MIL) or slide encoders for downstream tasks. However, patch-level SSL may overlook complex domain-specific features that are essential for biomarker prediction, such as mutation status and molecular characteristics, as SSL methods rely only on basic augmentations selected for natural image domains on small patch-level area. Moreover, SSL methods remain less data efficient than fully supervised approaches, requiring extensive computational resources and datasets to achieve competitive performance. To address these limitations, we present EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision. Using only 37k WSIs for training, EXAONE Path 2.0 achieves state-of-the-art average performance across 10 biomarker prediction tasks, demonstrating remarkable data efficiency. 

---
# Goal-Oriented Skill Abstraction for Offline Multi-Task Reinforcement Learning 

**Authors**: Jinmin He, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06628)  

**Abstract**: Offline multi-task reinforcement learning aims to learn a unified policy capable of solving multiple tasks using only pre-collected task-mixed datasets, without requiring any online interaction with the environment. However, it faces significant challenges in effectively sharing knowledge across tasks. Inspired by the efficient knowledge abstraction observed in human learning, we propose Goal-Oriented Skill Abstraction (GO-Skill), a novel approach designed to extract and utilize reusable skills to enhance knowledge transfer and task performance. Our approach uncovers reusable skills through a goal-oriented skill extraction process and leverages vector quantization to construct a discrete skill library. To mitigate class imbalances between broadly applicable and task-specific skills, we introduce a skill enhancement phase to refine the extracted skills. Furthermore, we integrate these skills using hierarchical policy learning, enabling the construction of a high-level policy that dynamically orchestrates discrete skills to accomplish specific tasks. Extensive experiments on diverse robotic manipulation tasks within the MetaWorld benchmark demonstrate the effectiveness and versatility of GO-Skill. 

---
# Q-STAC: Q-Guided Stein Variational Model Predictive Actor-Critic 

**Authors**: Shizhe Cai, Jayadeep Jacob, Zeya Yin, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2507.06625)  

**Abstract**: Deep reinforcement learning has shown remarkable success in continuous control tasks, yet often requires extensive training data, struggles with complex, long-horizon planning, and fails to maintain safety constraints during operation. Meanwhile, Model Predictive Control (MPC) offers explainability and constraint satisfaction, but typically yields only locally optimal solutions and demands careful cost function design. This paper introduces the Q-guided STein variational model predictive Actor-Critic (Q-STAC), a novel framework that bridges these approaches by integrating Bayesian MPC with actor-critic reinforcement learning through constrained Stein Variational Gradient Descent (SVGD). Our method optimizes control sequences directly using learned Q-values as objectives, eliminating the need for explicit cost function design while leveraging known system dynamics to enhance sample efficiency and ensure control signals remain within safe boundaries. Extensive experiments on 2D navigation and robotic manipulation tasks demonstrate that Q-STAC achieves superior sample efficiency, robustness, and optimality compared to state-of-the-art algorithms, while maintaining the high expressiveness of policy distributions. Experiment videos are available on our website: this https URL 

---
# Expediting data extraction using a large language model (LLM) and scoping review protocol: a methodological study within a complex scoping review 

**Authors**: James Stewart-Evans, Emma Wilson, Tessa Langley, Andrew Prayle, Angela Hands, Karen Exley, Jo Leonardi-Bee  

**Link**: [PDF](https://arxiv.org/pdf/2507.06623)  

**Abstract**: The data extraction stages of reviews are resource-intensive, and researchers may seek to expediate data extraction using online (large language models) LLMs and review protocols. Claude 3.5 Sonnet was used to trial two approaches that used a review protocol to prompt data extraction from 10 evidence sources included in a case study scoping review. A protocol-based approach was also used to review extracted data. Limited performance evaluation was undertaken which found high accuracy for the two extraction approaches (83.3% and 100%) when extracting simple, well-defined citation details; accuracy was lower (9.6% and 15.8%) when extracting more complex, subjective data items. Considering all data items, both approaches had precision >90% but low recall (<25%) and F1 scores (<40%). The context of a complex scoping review, open response types and methodological approach likely impacted performance due to missed and misattributed data. LLM feedback considered the baseline extraction accurate and suggested minor amendments: four of 15 (26.7%) to citation details and 8 of 38 (21.1%) to key findings data items were considered to potentially add value. However, when repeating the process with a dataset featuring deliberate errors, only 2 of 39 (5%) errors were detected. Review-protocol-based methods used for expediency require more robust performance evaluation across a range of LLMs and review contexts with comparison to conventional prompt engineering approaches. We recommend researchers evaluate and report LLM performance if using them similarly to conduct data extraction or review extracted data. LLM feedback contributed to protocol adaptation and may assist future review protocol drafting. 

---
# Efficient Multi-Task Reinforcement Learning with Cross-Task Policy Guidance 

**Authors**: Jinmin He, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06615)  

**Abstract**: Multi-task reinforcement learning endeavors to efficiently leverage shared information across various tasks, facilitating the simultaneous learning of multiple tasks. Existing approaches primarily focus on parameter sharing with carefully designed network structures or tailored optimization procedures. However, they overlook a direct and complementary way to exploit cross-task similarities: the control policies of tasks already proficient in some skills can provide explicit guidance for unmastered tasks to accelerate skills acquisition. To this end, we present a novel framework called Cross-Task Policy Guidance (CTPG), which trains a guide policy for each task to select the behavior policy interacting with the environment from all tasks' control policies, generating better training trajectories. In addition, we propose two gating mechanisms to improve the learning efficiency of CTPG: one gate filters out control policies that are not beneficial for guidance, while the other gate blocks tasks that do not necessitate guidance. CTPG is a general framework adaptable to existing parameter sharing approaches. Empirical evaluations demonstrate that incorporating CTPG with these approaches significantly enhances performance in manipulation and locomotion benchmarks. 

---
# Denoising Multi-Beta VAE: Representation Learning for Disentanglement and Generation 

**Authors**: Anshuk Uppal, Yuhta Takida, Chieh-Hsin Lai, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2507.06613)  

**Abstract**: Disentangled and interpretable latent representations in generative models typically come at the cost of generation quality. The $\beta$-VAE framework introduces a hyperparameter $\beta$ to balance disentanglement and reconstruction quality, where setting $\beta > 1$ introduces an information bottleneck that favors disentanglement over sharp, accurate reconstructions. To address this trade-off, we propose a novel generative modeling framework that leverages a range of $\beta$ values to learn multiple corresponding latent representations. First, we obtain a slew of representations by training a single variational autoencoder (VAE), with a new loss function that controls the information retained in each latent representation such that the higher $\beta$ value prioritize disentanglement over reconstruction fidelity. We then, introduce a non-linear diffusion model that smoothly transitions latent representations corresponding to different $\beta$ values. This model denoises towards less disentangled and more informative representations, ultimately leading to (almost) lossless representations, enabling sharp reconstructions. Furthermore, our model supports sample generation without input images, functioning as a standalone generative model. We evaluate our framework in terms of both disentanglement and generation quality. Additionally, we observe smooth transitions in the latent spaces with respect to changes in $\beta$, facilitating consistent manipulation of generated outputs. 

---
# Learning controllable dynamics through informative exploration 

**Authors**: Peter N. Loxley, Friedrich T. Sommer  

**Link**: [PDF](https://arxiv.org/pdf/2507.06582)  

**Abstract**: Environments with controllable dynamics are usually understood in terms of explicit models. However, such models are not always available, but may sometimes be learned by exploring an environment. In this work, we investigate using an information measure called "predicted information gain" to determine the most informative regions of an environment to explore next. Applying methods from reinforcement learning allows good suboptimal exploring policies to be found, and leads to reliable estimates of the underlying controllable dynamics. This approach is demonstrated by comparing with several myopic exploration approaches. 

---
# From Data-Centric to Sample-Centric: Enhancing LLM Reasoning via Progressive Optimization 

**Authors**: Xinjie Chen, Minpeng Liao, Guoxin Chen, Chengxi Li, Biao Fu, Kai Fan, Xinggao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06573)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has recently advanced the reasoning capabilities of large language models (LLMs). While prior work has emphasized algorithmic design, data curation, and reward shaping, we investigate RLVR from a sample-centric perspective and introduce LPPO (Learning-Progress and Prefix-guided Optimization), a framework of progressive optimization techniques. Our work addresses a critical question: how to best leverage a small set of trusted, high-quality demonstrations, rather than simply scaling up data volume. First, motivated by how hints aid human problem-solving, we propose prefix-guided sampling, an online data augmentation method that incorporates partial solution prefixes from expert demonstrations to guide the policy, particularly for challenging instances. Second, inspired by how humans focus on important questions aligned with their current capabilities, we introduce learning-progress weighting, a dynamic strategy that adjusts each training sample's influence based on model progression. We estimate sample-level learning progress via an exponential moving average of per-sample pass rates, promoting samples that foster learning and de-emphasizing stagnant ones. Experiments on mathematical-reasoning benchmarks demonstrate that our methods outperform strong baselines, yielding faster convergence and a higher performance ceiling. 

---
# SkyVLN: Vision-and-Language Navigation and NMPC Control for UAVs in Urban Environments 

**Authors**: Tianshun Li, Tianyi Huai, Zhen Li, Yichun Gao, Haoang Li, Xinhu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06564)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have emerged as versatile tools across various sectors, driven by their mobility and adaptability. This paper introduces SkyVLN, a novel framework integrating vision-and-language navigation (VLN) with Nonlinear Model Predictive Control (NMPC) to enhance UAV autonomy in complex urban environments. Unlike traditional navigation methods, SkyVLN leverages Large Language Models (LLMs) to interpret natural language instructions and visual observations, enabling UAVs to navigate through dynamic 3D spaces with improved accuracy and robustness. We present a multimodal navigation agent equipped with a fine-grained spatial verbalizer and a history path memory mechanism. These components allow the UAV to disambiguate spatial contexts, handle ambiguous instructions, and backtrack when necessary. The framework also incorporates an NMPC module for dynamic obstacle avoidance, ensuring precise trajectory tracking and collision prevention. To validate our approach, we developed a high-fidelity 3D urban simulation environment using AirSim, featuring realistic imagery and dynamic urban elements. Extensive experiments demonstrate that SkyVLN significantly improves navigation success rates and efficiency, particularly in new and unseen environments. 

---
# The Primacy of Magnitude in Low-Rank Adaptation 

**Authors**: Zicheng Zhang, Haoran Li, Yifeng Zhang, Guoqiang Gong, Jiaxing Wang, Pengzhang Liu, Qixia Jiang, Junxing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06558)  

**Abstract**: Low-Rank Adaptation (LoRA) offers a parameter-efficient paradigm for tuning large models. While recent spectral initialization methods improve convergence and performance over the naive "Noise & Zeros" scheme, their extra computational and storage overhead undermines efficiency. In this paper, we establish update magnitude as the fundamental driver of LoRA performance and propose LoRAM, a magnitude-driven "Basis & Basis" initialization scheme that matches spectral methods without their inefficiencies. Our key contributions are threefold: (i) Magnitude of weight updates determines convergence. We prove low-rank structures intrinsically bound update magnitudes, unifying hyperparameter tuning in learning rate, scaling factor, and initialization as mechanisms to optimize magnitude regulation. (ii) Spectral initialization succeeds via magnitude amplification. We demystify that the presumed knowledge-driven benefit of the spectral component essentially arises from the boost in the weight update magnitude. (iii) A novel and compact initialization strategy, LoRAM, scales deterministic orthogonal bases using pretrained weight magnitudes to simulate spectral gains. Extensive experiments show that LoRAM serves as a strong baseline, retaining the full efficiency of LoRA while matching or outperforming spectral initialization across benchmarks. 

---
# Graph-based Fake Account Detection: A Survey 

**Authors**: Ali Safarpoor Dehkordi, Ahad N. Zehmakan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06541)  

**Abstract**: In recent years, there has been a growing effort to develop effective and efficient algorithms for fake account detection in online social networks. This survey comprehensively reviews existing methods, with a focus on graph-based techniques that utilise topological features of social graphs (in addition to account information, such as their shared contents and profile data) to distinguish between fake and real accounts. We provide several categorisations of these methods (for example, based on techniques used, input data, and detection time), discuss their strengths and limitations, and explain how these methods connect in the broader context. We also investigate the available datasets, including both real-world data and synthesised models. We conclude the paper by proposing several potential avenues for future research. 

---
# InvestAlign: Overcoming Data Scarcity in Aligning Large Language Models with Investor Decision-Making Processes under Herd Behavior 

**Authors**: Huisheng Wang, Zhuoshi Pan, Hangjing Zhang, Mingxiao Liu, Hanqing Gao, H. Vicky Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06528)  

**Abstract**: Aligning Large Language Models (LLMs) with investor decision-making processes under herd behavior is a critical challenge in behavioral finance, which grapples with a fundamental limitation: the scarcity of real-user data needed for Supervised Fine-Tuning (SFT). While SFT can bridge the gap between LLM outputs and human behavioral patterns, its reliance on massive authentic data imposes substantial collection costs and privacy risks. We propose InvestAlign, a novel framework that constructs high-quality SFT datasets by leveraging theoretical solutions to similar and simple optimal investment problems rather than complex scenarios. Our theoretical analysis demonstrates that training LLMs with InvestAlign-generated data achieves faster parameter convergence than using real-user data, suggesting superior learning efficiency. Furthermore, we develop InvestAgent, an LLM agent fine-tuned with InvestAlign, which demonstrates significantly closer alignment to real-user data than pre-SFT models in both simple and complex investment problems. This highlights our proposed InvestAlign as a promising approach with the potential to address complex optimal investment problems and align LLMs with investor decision-making processes under herd behavior. Our code is publicly available at this https URL. 

---
# Gradientsys: A Multi-Agent LLM Scheduler with ReAct Orchestration 

**Authors**: Xinyuan Song, Zeyu Wang, Siyi Wu, Tianyu Shi, Lynn Ai  

**Link**: [PDF](https://arxiv.org/pdf/2507.06520)  

**Abstract**: We present Gradientsys, a next-generation multi-agent scheduling framework that coordinates diverse specialized AI agents using a typed Model-Context Protocol (MCP) and a ReAct-based dynamic planning loop. At its core, Gradientsys employs an LLM-powered scheduler for intelligent one-to-many task dispatch, enabling parallel execution of heterogeneous agents such as PDF parsers, web search modules, GUI controllers, and web builders. The framework supports hybrid synchronous/asynchronous execution, respects agent capacity constraints, and incorporates a robust retry-and-replan mechanism to handle failures gracefully. To promote transparency and trust, Gradientsys includes an observability layer streaming real-time agent activity and intermediate reasoning via Server-Sent Events (SSE). We offer an architectural overview and evaluate Gradientsys against existing frameworks in terms of extensibility, scheduling topology, tool reusability, parallelism, and observability. Experiments on the GAIA general-assistant benchmark show that Gradientsys achieves higher task success rates with reduced latency and lower API costs compared to a MinionS-style baseline, demonstrating the strength of its LLM-driven multi-agent orchestration. 

---
# Failure Forecasting Boosts Robustness of Sim2Real Rhythmic Insertion Policies 

**Authors**: Yuhan Liu, Xinyu Zhang, Haonan Chang, Abdeslam Boularias  

**Link**: [PDF](https://arxiv.org/pdf/2507.06519)  

**Abstract**: This paper addresses the challenges of Rhythmic Insertion Tasks (RIT), where a robot must repeatedly perform high-precision insertions, such as screwing a nut into a bolt with a wrench. The inherent difficulty of RIT lies in achieving millimeter-level accuracy and maintaining consistent performance over multiple repetitions, particularly when factors like nut rotation and friction introduce additional complexity. We propose a sim-to-real framework that integrates a reinforcement learning-based insertion policy with a failure forecasting module. By representing the wrench's pose in the nut's coordinate frame rather than the robot's frame, our approach significantly enhances sim-to-real transferability. The insertion policy, trained in simulation, leverages real-time 6D pose tracking to execute precise alignment, insertion, and rotation maneuvers. Simultaneously, a neural network predicts potential execution failures, triggering a simple recovery mechanism that lifts the wrench and retries the insertion. Extensive experiments in both simulated and real-world environments demonstrate that our method not only achieves a high one-time success rate but also robustly maintains performance over long-horizon repetitive tasks. 

---
# Towards LLM-based Root Cause Analysis of Hardware Design Failures 

**Authors**: Siyu Qiu, Muzhi Wang, Raheel Afsharmazayejani, Mohammad Moradi Shahmiri, Benjamin Tan, Hammond Pearce  

**Link**: [PDF](https://arxiv.org/pdf/2507.06512)  

**Abstract**: With advances in large language models (LLMs), new opportunities have emerged to develop tools that support the digital hardware design process. In this work, we explore how LLMs can assist with explaining the root cause of design issues and bugs that are revealed during synthesis and simulation, a necessary milestone on the pathway towards widespread use of LLMs in the hardware design process and for hardware security analysis. We find promising results: for our corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model reached a correct determination 100% of the time under pass@5 scoring, with other state of the art models and configurations usually achieving more than 80% performance and more than 90% when assisted with retrieval-augmented generation. 

---
# GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models 

**Authors**: Zhen Yang, Haitao Lin, Jiawei xue, Ziji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06507)  

**Abstract**: In the past year, Generative Recommendations (GRs) have undergone substantial advancements, especially in leveraging the powerful sequence modeling and reasoning capabilities of Large Language Models (LLMs) to enhance overall recommendation performance. LLM-based GRs are forming a new paradigm that is distinctly different from discriminative recommendations, showing strong potential to replace traditional recommendation systems heavily dependent on complex hand-crafted features. In this paper, we provide a comprehensive survey aimed at facilitating further research of LLM-based GRs. Initially, we outline the general preliminaries and application cases of LLM-based GRs. Subsequently, we introduce the main considerations when LLM-based GRs are applied in real industrial scenarios. Finally, we explore promising directions for LLM-based GRs. We hope that this survey contributes to the ongoing advancement of the GR domain. 

---
# Pun Intended: Multi-Agent Translation of Wordplay with Contrastive Learning and Phonetic-Semantic Embeddings 

**Authors**: Russell Taylor, Benjamin Herbert, Michael Sana  

**Link**: [PDF](https://arxiv.org/pdf/2507.06506)  

**Abstract**: Translating wordplay across languages presents unique challenges that have long confounded both professional human translators and machine translation systems. This research proposes a novel approach for translating puns from English to French by combining state-of-the-art large language models with specialized techniques for wordplay generation.
Our methodology employs a three-stage approach. First, we establish a baseline using multiple frontier large language models with feedback based on a new contrastive learning dataset. Second, we implement a guided chain-of-thought pipeline with combined phonetic-semantic embeddings. Third, we implement a multi-agent generator-discriminator framework for evaluating and regenerating puns with feedback.
Moving beyond the limitations of literal translation, our methodology's primary objective is to capture the linguistic creativity and humor of the source text wordplay, rather than simply duplicating its vocabulary. Our best runs earned first and second place in the CLEF JOKER 2025 Task 2 competition where they were evaluated manually by expert native French speakers.
This research addresses a gap between translation studies and computational linguistics by implementing linguistically-informed techniques for wordplay translation, advancing our understanding of how language models can be leveraged to handle the complex interplay between semantic ambiguity, phonetic similarity, and the implicit cultural and linguistic awareness needed for successful humor. 

---
# MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models 

**Authors**: Yiwen Liu, Chenyu Zhang, Junjie Song, Siqi Chen, Sun Yin, Zihan Wang, Lingming Zeng, Yuji Cao, Junming Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06502)  

**Abstract**: As a prominent data modality task, time series forecasting plays a pivotal role in diverse applications. With the remarkable advancements in Large Language Models (LLMs), the adoption of LLMs as the foundational architecture for time series modeling has gained significant attention. Although existing models achieve some success, they rarely both model time and frequency characteristics in a pretraining-finetuning paradigm leading to suboptimal performance in predictions of complex time series, which requires both modeling periodicity and prior pattern knowledge of signals. We propose MoFE-Time, an innovative time series forecasting model that integrates time and frequency domain features within a Mixture of Experts (MoE) network. Moreover, we use the pretraining-finetuning paradigm as our training framework to effectively transfer prior pattern knowledge across pretraining and finetuning datasets with different periodicity distributions. Our method introduces both frequency and time cells as experts after attention modules and leverages the MoE routing mechanism to construct multidimensional sparse representations of input signals. In experiments on six public benchmarks, MoFE-Time has achieved new state-of-the-art performance, reducing MSE and MAE by 6.95% and 6.02% compared to the representative methods Time-MoE. Beyond the existing evaluation benchmarks, we have developed a proprietary dataset, NEV-sales, derived from real-world business scenarios. Our method achieves outstanding results on this dataset, underscoring the effectiveness of the MoFE-Time model in practical commercial applications. 

---
# Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning 

**Authors**: Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2507.06485)  

**Abstract**: Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and finetuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Based on observations about the data scaling of RL samples, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by an average of 2.4% in accuracy using only 3.6% training samples. For example, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark, and a 2.6% improvement on MMVU. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance. 

---
# Generative Lagrangian data assimilation for ocean dynamics under extreme sparsity 

**Authors**: Niloofar Asefi, Leonard Lupin-Jimenez, Tianning Wu, Ruoying He, Ashesh Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2507.06479)  

**Abstract**: Reconstructing ocean dynamics from observational data is fundamentally limited by the sparse, irregular, and Lagrangian nature of spatial sampling, particularly in subsurface and remote regions. This sparsity poses significant challenges for forecasting key phenomena such as eddy shedding and rogue waves. Traditional data assimilation methods and deep learning models often struggle to recover mesoscale turbulence under such constraints. We leverage a deep learning framework that combines neural operators with denoising diffusion probabilistic models (DDPMs) to reconstruct high-resolution ocean states from extremely sparse Lagrangian observations. By conditioning the generative model on neural operator outputs, the framework accurately captures small-scale, high-wavenumber dynamics even at $99\%$ sparsity (for synthetic data) and $99.9\%$ sparsity (for real satellite observations). We validate our method on benchmark systems, synthetic float observations, and real satellite data, demonstrating robust performance under severe spatial sampling limitations as compared to other deep learning baselines. 

---
# Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models 

**Authors**: Aaron Dharna, Cong Lu, Jeff Clune  

**Link**: [PDF](https://arxiv.org/pdf/2507.06466)  

**Abstract**: Multi-agent interactions have long fueled innovation, from natural predator-prey dynamics to the space race. Self-play (SP) algorithms try to harness these dynamics by pitting agents against ever-improving opponents, thereby creating an implicit curriculum toward learning high-quality solutions. However, SP often fails to produce diverse solutions and can get stuck in locally optimal behaviors. We introduce Foundation-Model Self-Play (FMSP), a new direction that leverages the code-generation capabilities and vast knowledge of foundation models (FMs) to overcome these challenges by leaping across local optima in policy space. We propose a family of approaches: (1) \textbf{Vanilla Foundation-Model Self-Play (vFMSP)} continually refines agent policies via competitive self-play; (2) \textbf{Novelty-Search Self-Play (NSSP)} builds a diverse population of strategies, ignoring performance; and (3) the most promising variant, \textbf{Quality-Diveristy Self-Play (QDSP)}, creates a diverse set of high-quality policies by combining the diversity of NSSP and refinement of vFMSP. We evaluate FMSPs in Car Tag, a continuous-control pursuer-evader setting, and in Gandalf, a simple AI safety simulation in which an attacker tries to jailbreak an LLM's defenses. In Car Tag, FMSPs explore a wide variety of reinforcement learning, tree search, and heuristic-based methods, to name just a few. In terms of discovered policy quality, \ouralgo and vFMSP surpass strong human-designed strategies. In Gandalf, FMSPs can successfully automatically red-team an LLM, breaking through and jailbreaking six different, progressively stronger levels of defense. Furthermore, FMSPs can automatically proceed to patch the discovered vulnerabilities. Overall, FMSPs represent a promising new research frontier of improving self-play with foundation models, opening fresh paths toward more creative and open-ended strategy discovery 

---
# SoftSignSGD(S3): An Enhanced Optimizer for Practical DNN Training and Loss Spikes Minimization Beyond Adam 

**Authors**: Hanyang Peng, Shuang Qin, Yue Yu, Fangqing Jiang, Hui Wang, Wen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06464)  

**Abstract**: Adam has proven remarkable successful in training deep neural networks, but the mechanisms underlying its empirical successes and limitations remain underexplored. In this study, we demonstrate that the effectiveness of Adam stems largely from its similarity to SignSGD in robustly handling large gradient fluctuations, yet it is also vulnerable to destabilizing loss spikes due to its uncontrolled update scaling. To enhance the advantage of Adam and mitigate its limitation, we propose SignSoftSGD (S3), a novel optimizer with three key innovations. \emph{First}, S3 generalizes the sign-like update by employing a flexible $p$-th order momentum ($p \geq 1$) in the denominator, departing from the conventional second-order momentum (variance) preconditioning. This design enables enhanced performance while achieving stable training even with aggressive learning rates. \emph{Second}, S3 minimizes the occurrences of loss spikes through unified exponential moving average coefficients for numerator and denominator momenta, which inherently bound updates to $[-1, 1]$ and simplify hyperparameter tuning. \emph{Third}, S3 incorporates an equivalent Nesterov's accelerated gradient(NAG) module, accelerating convergence without memory overhead. Theoretically, we prove that S3 achieves the optimal convergence rate of $O\left(\frac{1}{T^{\sfrac{1}{4}}}\right)$ for general nonconvex stochastic optimization under weak assumptions. Extensive experiments across a range of vision and language tasks show that \textsf{\small S3} not only converges more rapidly and improves performance but also rarely experiences loss spikes, even with a \textbf{$\bm{10 \times}$} larger learning rate. In fact, S3 delivers performance comparable to or better than AdamW with \textbf{$2 \times$} the training steps, establishing its efficacy in both efficiency and final task performance. 

---
# EA: An Event Autoencoder for High-Speed Vision Sensing 

**Authors**: Riadul Islam, Joey Mulé, Dhandeep Challagundla, Shahmir Rizvi, Sean Carson  

**Link**: [PDF](https://arxiv.org/pdf/2507.06459)  

**Abstract**: High-speed vision sensing is essential for real-time perception in applications such as robotics, autonomous vehicles, and industrial automation. Traditional frame-based vision systems suffer from motion blur, high latency, and redundant data processing, limiting their performance in dynamic environments. Event cameras, which capture asynchronous brightness changes at the pixel level, offer a promising alternative but pose challenges in object detection due to sparse and noisy event streams. To address this, we propose an event autoencoder architecture that efficiently compresses and reconstructs event data while preserving critical spatial and temporal features. The proposed model employs convolutional encoding and incorporates adaptive threshold selection and a lightweight classifier to enhance recognition accuracy while reducing computational complexity. Experimental results on the existing Smart Event Face Dataset (SEFD) demonstrate that our approach achieves comparable accuracy to the YOLO-v4 model while utilizing up to $35.5\times$ fewer parameters. Implementations on embedded platforms, including Raspberry Pi 4B and NVIDIA Jetson Nano, show high frame rates ranging from 8 FPS up to 44.8 FPS. The proposed classifier exhibits up to 87.84x better FPS than the state-of-the-art and significantly improves event-based vision performance, making it ideal for low-power, high-speed applications in real-time edge computing. 

---
# FedPhD: Federated Pruning with Hierarchical Learning of Diffusion Models 

**Authors**: Qianyu Long, Qiyuan Wang, Christos Anagnostopoulos, Daning Bi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06449)  

**Abstract**: Federated Learning (FL), as a distributed learning paradigm, trains models over distributed clients' data. FL is particularly beneficial for distributed training of Diffusion Models (DMs), which are high-quality image generators that require diverse data. However, challenges such as high communication costs and data heterogeneity persist in training DMs similar to training Transformers and Convolutional Neural Networks. Limited research has addressed these issues in FL environments. To address this gap and challenges, we introduce a novel approach, FedPhD, designed to efficiently train DMs in FL environments. FedPhD leverages Hierarchical FL with homogeneity-aware model aggregation and selection policy to tackle data heterogeneity while reducing communication costs. The distributed structured pruning of FedPhD enhances computational efficiency and reduces model storage requirements in clients. Our experiments across multiple datasets demonstrate that FedPhD achieves high model performance regarding Fréchet Inception Distance (FID) scores while reducing communication costs by up to $88\%$. FedPhD outperforms baseline methods achieving at least a $34\%$ improvement in FID, while utilizing only $56\%$ of the total computation and communication resources. 

---
# Can Interpretation Predict Behavior on Unseen Data? 

**Authors**: Victoria R. Li, Jenny Kaufmann, Martin Wattenberg, David Alvarez-Melis, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2507.06445)  

**Abstract**: Interpretability research often aims to predict how a model will respond to targeted interventions on specific mechanisms. However, it rarely predicts how a model will respond to unseen input data. This paper explores the promises and challenges of interpretability as a tool for predicting out-of-distribution (OOD) model behavior. Specifically, we investigate the correspondence between attention patterns and OOD generalization in hundreds of Transformer models independently trained on a synthetic classification task. These models exhibit several distinct systematic generalization rules OOD, forming a diverse population for correlational analysis. In this setting, we find that simple observational tools from interpretability can predict OOD performance. In particular, when in-distribution attention exhibits hierarchical patterns, the model is likely to generalize hierarchically on OOD data -- even when the rule's implementation does not rely on these hierarchical patterns, according to ablation tests. Our findings offer a proof-of-concept to motivate further interpretability work on predicting unseen model behavior. 

---
# Assessing the Prevalence of AI-assisted Cheating in Programming Courses: A Pilot Study 

**Authors**: Kaléu Delphino  

**Link**: [PDF](https://arxiv.org/pdf/2507.06438)  

**Abstract**: Tools that can generate computer code in response to inputs written in natural language, such as ChatGPT, pose an existential threat to Computer Science education in its current form, since students can now use these tools to solve assignments without much effort. While that risk has already been recognized by scholars, the proportion of the student body that is incurring in this new kind of plagiarism is still an open problem. We conducted a pilot study in a large CS class (n=120) to assess the feasibility of estimating AI plagiarism through anonymous surveys and interviews. More than 25% of the survey respondents admitted to committing AI plagiarism. Conversely, only one student accepted to be interviewed. Given the high levels of misconduct acknowledgment, we conclude that surveys are an effective method for studies on the matter, while interviews should be avoided or designed in a way that can entice participation. 

---
# Deprecating Benchmarks: Criteria and Framework 

**Authors**: Ayrton San Joaquin, Rokas Gipiškis, Leon Staufer, Ariel Gil  

**Link**: [PDF](https://arxiv.org/pdf/2507.06434)  

**Abstract**: As frontier artificial intelligence (AI) models rapidly advance, benchmarks are integral to comparing different models and measuring their progress in different task-specific domains. However, there is a lack of guidance on when and how benchmarks should be deprecated once they cease to effectively perform their purpose. This risks benchmark scores over-valuing model capabilities, or worse, obscuring capabilities and safety-washing. Based on a review of benchmarking practices, we propose criteria to decide when to fully or partially deprecate benchmarks, and a framework for deprecating benchmarks. Our work aims to advance the state of benchmarking towards rigorous and quality evaluations, especially for frontier models, and our recommendations are aimed to benefit benchmark developers, benchmark users, AI governance actors (across governments, academia, and industry panels), and policy makers. 

---
# Bridging Data Gaps of Rare Conditions in ICU: A Multi-Disease Adaptation Approach for Clinical Prediction 

**Authors**: Mingcheng Zhu, Yu Liu, Zhiyao Luo, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06432)  

**Abstract**: Artificial Intelligence has revolutionised critical care for common conditions. Yet, rare conditions in the intensive care unit (ICU), including recognised rare diseases and low-prevalence conditions in the ICU, remain underserved due to data scarcity and intra-condition heterogeneity. To bridge such gaps, we developed KnowRare, a domain adaptation-based deep learning framework for predicting clinical outcomes for rare conditions in the ICU. KnowRare mitigates data scarcity by initially learning condition-agnostic representations from diverse electronic health records through self-supervised pre-training. It addresses intra-condition heterogeneity by selectively adapting knowledge from clinically similar conditions with a developed condition knowledge graph. Evaluated on two ICU datasets across five clinical prediction tasks (90-day mortality, 30-day readmission, ICU mortality, remaining length of stay, and phenotyping), KnowRare consistently outperformed existing state-of-the-art models. Additionally, KnowRare demonstrated superior predictive performance compared to established ICU scoring systems, including APACHE IV and IV-a. Case studies further demonstrated KnowRare's flexibility in adapting its parameters to accommodate dataset-specific and task-specific characteristics, its generalisation to common conditions under limited data scenarios, and its rationality in selecting source conditions. These findings highlight KnowRare's potential as a robust and practical solution for supporting clinical decision-making and improving care for rare conditions in the ICU. 

---
# SImpHAR: Advancing impedance-based human activity recognition using 3D simulation and text-to-motion models 

**Authors**: Lala Shakti Swarup Ray, Mengxi Liu, Deepika Gurung, Bo Zhou, Sungho Suh, Paul Lukowicz  

**Link**: [PDF](https://arxiv.org/pdf/2507.06405)  

**Abstract**: Human Activity Recognition (HAR) with wearable sensors is essential for applications in healthcare, fitness, and human-computer interaction. Bio-impedance sensing offers unique advantages for fine-grained motion capture but remains underutilized due to the scarcity of labeled data. We introduce SImpHAR, a novel framework addressing this limitation through two core contributions. First, we propose a simulation pipeline that generates realistic bio-impedance signals from 3D human meshes using shortest-path estimation, soft-body physics, and text-to-motion generation serving as a digital twin for data augmentation. Second, we design a two-stage training strategy with decoupled approach that enables broader activity coverage without requiring label-aligned synthetic data. We evaluate SImpHAR on our collected ImpAct dataset and two public benchmarks, showing consistent improvements over state-of-the-art methods, with gains of up to 22.3% and 21.8%, in terms of accuracy and macro F1 score, respectively. Our results highlight the promise of simulation-driven augmentation and modular training for impedance-based HAR. 

---
# An AI-Driven Thermal-Fluid Testbed for Advanced Small Modular Reactors: Integration of Digital Twin and Large Language Models 

**Authors**: Doyeong Lim, Yang Liu, Zavier Ndum Ndum, Christian Young, Yassin Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06399)  

**Abstract**: This paper presents a multipurpose artificial intelligence (AI)-driven thermal-fluid testbed designed to advance Small Modular Reactor technologies by seamlessly integrating physical experimentation with advanced computational intelligence. The platform uniquely combines a versatile three-loop thermal-fluid facility with a high-fidelity digital twin and sophisticated AI frameworks for real-time prediction, control, and operational assistance. Methodologically, the testbed's digital twin, built upon the System Analysis Module code, is coupled with a Gated Recurrent Unit (GRU) neural network. This machine learning model, trained on experimental data, enables faster-than-real-time simulation, providing predictive insights into the system's dynamic behavior. The practical application of this AI integration is showcased through case studies. An AI-driven control framework where the GRU model accurately forecasts future system states and the corresponding control actions required to meet operational demands. Furthermore, an intelligent assistant, powered by a large language model, translates complex sensor data and simulation outputs into natural language, offering operators actionable analysis and safety recommendations. Comprehensive validation against experimental transients confirms the platform's high fidelity, with the GRU model achieving a temperature prediction root mean square error of 1.42 K. This work establishes an integrated research environment at the intersection of AI and thermal-fluid science, showcasing how AI-driven methodologies in modeling, control, and operator support can accelerate the innovation and deployment of next-generation nuclear systems. 

---
# KPFlow: An Operator Perspective on Dynamic Collapse Under Gradient Descent Training of Recurrent Networks 

**Authors**: James Hazelden, Laura Driscoll, Eli Shlizerman, Eric Shea-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2507.06381)  

**Abstract**: Gradient Descent (GD) and its variants are the primary tool for enabling efficient training of recurrent dynamical systems such as Recurrent Neural Networks (RNNs), Neural ODEs and Gated Recurrent units (GRUs). The dynamics that are formed in these models exhibit features such as neural collapse and emergence of latent representations that may support the remarkable generalization properties of networks. In neuroscience, qualitative features of these representations are used to compare learning in biological and artificial systems. Despite recent progress, there remains a need for theoretical tools to rigorously understand the mechanisms shaping learned representations, especially in finite, non-linear models. Here, we show that the gradient flow, which describes how the model's dynamics evolve over GD, can be decomposed into a product that involves two operators: a Parameter Operator, K, and a Linearized Flow Propagator, P. K mirrors the Neural Tangent Kernel in feed-forward neural networks, while P appears in Lyapunov stability and optimal control theory. We demonstrate two applications of our decomposition. First, we show how their interplay gives rise to low-dimensional latent dynamics under GD, and, specifically, how the collapse is a result of the network structure, over and above the nature of the underlying task. Second, for multi-task training, we show that the operators can be used to measure how objectives relevant to individual sub-tasks align. We experimentally and theoretically validate these findings, providing an efficient Pytorch package, \emph{KPFlow}, implementing robust analysis tools for general recurrent architectures. Taken together, our work moves towards building a next stage of understanding of GD learning in non-linear recurrent models. 

---
# Secure and Storage-Efficient Deep Learning Models for Edge AI Using Automatic Weight Generation 

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia  

**Link**: [PDF](https://arxiv.org/pdf/2507.06380)  

**Abstract**: Complex neural networks require substantial memory to store a large number of synaptic weights. This work introduces WINGs (Automatic Weight Generator for Secure and Storage-Efficient Deep Learning Models), a novel framework that dynamically generates layer weights in a fully connected neural network (FC) and compresses the weights in convolutional neural networks (CNNs) during inference, significantly reducing memory requirements without sacrificing accuracy. WINGs framework uses principal component analysis (PCA) for dimensionality reduction and lightweight support vector regression (SVR) models to predict layer weights in the FC networks, removing the need for storing full-weight matrices and achieving substantial memory savings. It also preferentially compresses the weights in low-sensitivity layers of CNNs using PCA and SVR with sensitivity analysis. The sensitivity-aware design also offers an added level of security, as any bit-flip attack with weights in compressed layers has an amplified and readily detectable effect on accuracy. WINGs achieves 53x compression for the FC layers and 28x for AlexNet with MNIST dataset, and 18x for Alexnet with CIFAR-10 dataset with 1-2% accuracy loss. This significant reduction in memory results in higher throughput and lower energy for DNN inference, making it attractive for resource-constrained edge applications. 

---
# SymFlux: deep symbolic regression of Hamiltonian vector fields 

**Authors**: M.A. Evangelista-Alvarado, P. Suárez-Serrato  

**Link**: [PDF](https://arxiv.org/pdf/2507.06342)  

**Abstract**: We present SymFlux, a novel deep learning framework that performs symbolic regression to identify Hamiltonian functions from their corresponding vector fields on the standard symplectic plane. SymFlux models utilize hybrid CNN-LSTM architectures to learn and output the symbolic mathematical expression of the underlying Hamiltonian. Training and validation are conducted on newly developed datasets of Hamiltonian vector fields, a key contribution of this work. Our results demonstrate the model's effectiveness in accurately recovering these symbolic expressions, advancing automated discovery in Hamiltonian mechanics. 

---
# MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing 

**Authors**: Michael Clemens, Ana Marasović  

**Link**: [PDF](https://arxiv.org/pdf/2507.06329)  

**Abstract**: While AI presents significant potential for enhancing music mixing and mastering workflows, current research predominantly emphasizes end-to-end automation or generation, often overlooking the collaborative and instructional dimensions vital for co-creative processes. This gap leaves artists, particularly amateurs seeking to develop expertise, underserved. To bridge this, we introduce MixAssist, a novel audio-language dataset capturing the situated, multi-turn dialogue between expert and amateur music producers during collaborative mixing sessions. Comprising 431 audio-grounded conversational turns derived from 7 in-depth sessions involving 12 producers, MixAssist provides a unique resource for training and evaluating audio-language models that can comprehend and respond to the complexities of real-world music production dialogues. Our evaluations, including automated LLM-as-a-judge assessments and human expert comparisons, demonstrate that fine-tuning models such as Qwen-Audio on MixAssist can yield promising results, with Qwen significantly outperforming other tested models in generating helpful, contextually relevant mixing advice. By focusing on co-creative instruction grounded in audio context, MixAssist enables the development of intelligent AI assistants designed to support and augment the creative process in music mixing. 

---
# Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson's Disease 

**Authors**: Harsh Ravivarapu, Gaurav Bagwe, Xiaoyong Yuan, Chunxiu Yu, Lan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06326)  

**Abstract**: Deep brain stimulation (DBS) is an established intervention for Parkinson's disease (PD), but conventional open-loop systems lack adaptability, are energy-inefficient due to continuous stimulation, and provide limited personalization to individual neural dynamics. Adaptive DBS (aDBS) offers a closed-loop alternative, using biomarkers such as beta-band oscillations to dynamically modulate stimulation. While reinforcement learning (RL) holds promise for personalized aDBS control, existing methods suffer from high sample complexity, unstable exploration in binary action spaces, and limited deployability on resource-constrained hardware.
We propose SEA-DBS, a sample-efficient actor-critic framework that addresses the core challenges of RL-based adaptive neurostimulation. SEA-DBS integrates a predictive reward model to reduce reliance on real-time feedback and employs Gumbel Softmax-based exploration for stable, differentiable policy updates in binary action spaces. Together, these components improve sample efficiency, exploration robustness, and compatibility with resource-constrained neuromodulatory hardware. We evaluate SEA-DBS on a biologically realistic simulation of Parkinsonian basal ganglia activity, demonstrating faster convergence, stronger suppression of pathological beta-band power, and resilience to post-training FP16 quantization. Our results show that SEA-DBS offers a practical and effective RL-based aDBS framework for real-time, resource-constrained neuromodulation. 

---
# Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms 

**Authors**: Tarek Gasmi, Ramzi Guesmi, Ines Belhadj, Jihene Bennaceur  

**Link**: [PDF](https://arxiv.org/pdf/2507.06323)  

**Abstract**: Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security. 

---
# Too Human to Model:The Uncanny Valley of LLMs in Social Simulation -- When Generative Language Agents Misalign with Modelling Principles 

**Authors**: Yongchao Zeng, Calum Brown, Mark Rounsevell  

**Link**: [PDF](https://arxiv.org/pdf/2507.06310)  

**Abstract**: Large language models (LLMs) have been increasingly used to build agents in social simulation because of their impressive abilities to generate fluent, contextually coherent dialogues. Such abilities can enhance the realism of models. However, the pursuit of realism is not necessarily compatible with the epistemic foundation of modelling. We argue that LLM agents, in many regards, are too human to model: they are too expressive, detailed and intractable to be consistent with the abstraction, simplification, and interpretability typically demanded by modelling. Through a model-building thought experiment that converts the Bass diffusion model to an LLM-based variant, we uncover five core dilemmas: a temporal resolution mismatch between natural conversation and abstract time steps; the need for intervention in conversations while avoiding undermining spontaneous agent outputs; the temptation to introduce rule-like instructions in prompts while maintaining conversational naturalness; the tension between role consistency and role evolution across time; and the challenge of understanding emergence, where system-level patterns become obscured by verbose micro textual outputs. These dilemmas steer the LLM agents towards an uncanny valley: not abstract enough to clarify underlying social mechanisms, while not natural enough to represent realistic human behaviour. This exposes an important paradox: the realism of LLM agents can obscure, rather than clarify, social dynamics when misapplied. We tease out the conditions in which LLM agents are ideally suited: where system-level emergence is not the focus, linguistic nuances and meaning are central, interactions unfold in natural time, and stable role identity is more important than long-term behavioural evolution. We call for repositioning LLM agents in the ecosystem of social simulation for future applications. 

---
# Humans overrely on overconfident language models, across languages 

**Authors**: Neil Rathi, Dan Jurafsky, Kaitlyn Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06306)  

**Abstract**: As large language models (LLMs) are deployed globally, it is crucial that their responses are calibrated across languages to accurately convey uncertainty and limitations. Previous work has shown that LLMs are linguistically overconfident in English, leading users to overrely on confident generations. However, the usage and interpretation of epistemic markers (e.g., 'It's definitely,' 'I think') can differ sharply across languages. Here, we study the risks of multilingual linguistic (mis)calibration, overconfidence, and overreliance across five languages to evaluate the safety of LLMs in a global context.
We find that overreliance risks are high across all languages. We first analyze the distribution of LLM-generated epistemic markers, and observe that while LLMs are cross-linguistically overconfident, they are also sensitive to documented linguistic variation. For example, models generate the most markers of uncertainty in Japanese and the most markers of certainty in German and Mandarin. We then measure human reliance rates across languages, finding that while users strongly rely on confident LLM generations in all languages, reliance behaviors differ cross-linguistically: for example, users rely significantly more on expressions of uncertainty in Japanese than in English. Taken together, these results indicate high risk of reliance on overconfident model generations across languages. Our findings highlight the challenges of multilingual linguistic calibration and stress the importance of culturally and linguistically contextualized model safety evaluations. 

---
# The bitter lesson of misuse detection 

**Authors**: Hadrien Mariaccia, Charbel-Raphaël Segerie, Diego Dorn  

**Link**: [PDF](https://arxiv.org/pdf/2507.06282)  

**Abstract**: Prior work on jailbreak detection has established the importance of adversarial robustness for LLMs but has largely focused on the model ability to resist adversarial inputs and to output safe content, rather than the effectiveness of external supervision systems. The only public and independent benchmark of these guardrails to date evaluates a narrow set of supervisors on limited scenarios. Consequently, no comprehensive public benchmark yet verifies how well supervision systems from the market perform under realistic, diverse attacks. To address this, we introduce BELLS, a Benchmark for the Evaluation of LLM Supervision Systems. The framework is two dimensional: harm severity (benign, borderline, harmful) and adversarial sophistication (direct vs. jailbreak) and provides a rich dataset covering 3 jailbreak families and 11 harm categories. Our evaluations reveal drastic limitations of specialized supervision systems. While they recognize some known jailbreak patterns, their semantic understanding and generalization capabilities are very limited, sometimes with detection rates close to zero when asking a harmful question directly or with a new jailbreak technique such as base64 encoding. Simply asking generalist LLMs if the user question is "harmful or not" largely outperforms these supervisors from the market according to our BELLS score. But frontier LLMs still suffer from metacognitive incoherence, often responding to queries they correctly identify as harmful (up to 30 percent for Claude 3.7 and greater than 50 percent for Mistral Large). These results suggest that simple scaffolding could significantly improve misuse detection robustness, but more research is needed to assess the tradeoffs of such techniques. Our results support the "bitter lesson" of misuse detection: general capabilities of LLMs are necessary to detect a diverse array of misuses and jailbreaks. 

---
# A Survey of Multi Agent Reinforcement Learning: Federated Learning and Cooperative and Noncooperative Decentralized Regimes 

**Authors**: Kemboi Cheruiyot, Nickson Kiprotich, Vyacheslav Kungurtsev, Kennedy Mugo, Vivian Mwirigi, Marvin Ngesa  

**Link**: [PDF](https://arxiv.org/pdf/2507.06278)  

**Abstract**: The increasing interest in research and innovation towards the development of autonomous agents presents a number of complex yet important scenarios of multiple AI Agents interacting with each other in an environment. The particular setting can be understood as exhibiting three possibly topologies of interaction - centrally coordinated cooperation, ad-hoc interaction and cooperation, and settings with noncooperative incentive structures. This article presents a comprehensive survey of all three domains, defined under the formalism of Federal Reinforcement Learning (RL), Decentralized RL, and Noncooperative RL, respectively. Highlighting the structural similarities and distinctions, we review the state of the art in these subjects, primarily explored and developed only recently in the literature. We include the formulations as well as known theoretical guarantees and highlights and limitations of numerical performance. 

---
# The Prompt War: How AI Decides on a Military Intervention 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2507.06277)  

**Abstract**: Which factors determine AI propensity for military intervention? While the use of AI in war games and military planning is growing exponentially, the simple analysis of key drivers embedded in the models has not yet been done. This paper does a simple conjoint experiment proposing a model to decide on military intervention in 640 vignettes where each was run for 100 times allowing to explore AI decision on military intervention systematically. The analysis finds that largest predictors of AI decision to intervene are high domestic support and high probability of success. Costs such as international condemnation, military deaths, civilian deaths, and negative economic effect are statistically significant, but their effect is around half of domestic support and probability of victory. Closing window of opportunity only reaches statistical significance in interaction with other factors. The results are remarkably consistent across scenarios and across different models (OpenAI GPT, Anthropic Claude, Google Gemini) suggesting a pattern in AI decision-making. 

---
# Advancing Offline Handwritten Text Recognition: A Systematic Review of Data Augmentation and Generation Techniques 

**Authors**: Yassin Hussein Rassul, Aram M. Ahmed, Polla Fattah, Bryar A. Hassan, Arwaa W. Abdulkareem, Tarik A. Rashid, Joan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06275)  

**Abstract**: Offline Handwritten Text Recognition (HTR) systems play a crucial role in applications such as historical document digitization, automatic form processing, and biometric authentication. However, their performance is often hindered by the limited availability of annotated training data, particularly for low-resource languages and complex scripts. This paper presents a comprehensive survey of offline handwritten data augmentation and generation techniques designed to improve the accuracy and robustness of HTR systems. We systematically examine traditional augmentation methods alongside recent advances in deep learning, including Generative Adversarial Networks (GANs), diffusion models, and transformer-based approaches. Furthermore, we explore the challenges associated with generating diverse and realistic handwriting samples, particularly in preserving script authenticity and addressing data scarcity. This survey follows the PRISMA methodology, ensuring a structured and rigorous selection process. Our analysis began with 1,302 primary studies, which were filtered down to 848 after removing duplicates, drawing from key academic sources such as IEEE Digital Library, Springer Link, Science Direct, and ACM Digital Library. By evaluating existing datasets, assessment metrics, and state-of-the-art methodologies, this survey identifies key research gaps and proposes future directions to advance the field of handwritten text generation across diverse linguistic and stylistic landscapes. 

---
# Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks 

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06274)  

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings. 

---
# Magneto-radiative modelling and artificial neural network optimization of biofluid flow in a stenosed arterial domain 

**Authors**: S P Shivakumar, Gunisetty Ramasekhar, P Nimmy, Sujesh Areekara, L Thanuja, T V Smitha, S Devanathan, Ganesh R Naik, K V Nagaraja  

**Link**: [PDF](https://arxiv.org/pdf/2507.06273)  

**Abstract**: The increasing complexity of cardiovascular diseases and limitations in traditional healing methods mandate the invention of new drug delivery systems that assure targeted, effective, and regulated treatments, contributing directly to UN SDGs 3 and 9, thereby encouraging the utilization of sustainable medical technologies in healthcare. This study investigates the flow of a Casson-Maxwell nanofluid through a stenosed arterial domain. The quantities, such as skin friction and heat transfer rate, are analysed in detail. The Casson-Maxwell fluid shows a lower velocity profile than the Casson fluids, which indicates the improved residence time for efficient drug delivery. The heat transfer rate shows an increase with higher volume fractions of copper and aluminium oxide nanoparticles and a decrease with higher volume fractions of silver nanoparticles. The skin friction coefficient decreases by 219% with a unit increase in the Maxwell parameter, whereas it increases by 66.1% with a unit rise in the Casson parameter. This work supports SDGs 4 and 17 by fostering interdisciplinary learning and collaboration in fluid dynamics and healthcare innovation. Additionally, the rate of heat flow was forecasted (with an overall R-value of 0.99457) using the Levenberg-Marquardt backpropagation training scheme under the influence of magneto-radiative, linear heat source and Casson-Maxwell parameters along with the tri-metallic nanoparticle volume fractions. It is also observed that the drag coefficient is most sensitive to the changes in the Maxwell parameter. 

---
# LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance 

**Authors**: Zhang Li, Biao Yang, Qiang Liu, Shuo Zhang, Zhiyin Ma, Shuo Zhang, Liang Yin, Linger Deng, Yabo Sun, Yuliang Liu, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.06272)  

**Abstract**: While large multi-modal models (LMMs) demonstrate promising capabilities in segmentation and comprehension, they still struggle with two limitations: inaccurate segmentation and hallucinated comprehension. These challenges stem primarily from constraints in weak visual comprehension and a lack of fine-grained perception. To alleviate these limitations, we propose LIRA, a framework that capitalizes on the complementary relationship between visual comprehension and segmentation via two key components: (1) Semantic-Enhanced Feature Extractor (SEFE) improves object attribute inference by fusing semantic and pixel-level features, leading to more accurate segmentation; (2) Interleaved Local Visual Coupling (ILVC) autoregressively generates local descriptions after extracting local features based on segmentation masks, offering fine-grained supervision to mitigate hallucinations. Furthermore, we find that the precision of object segmentation is positively correlated with the latent related semantics of the <seg> token. To quantify this relationship and the model's potential semantic inferring ability, we introduce the Attributes Evaluation (AttrEval) dataset. Our experiments show that LIRA achieves state-of-the-art performance in both segmentation and comprehension tasks. Code will be available at this https URL. 

---
# A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry 

**Authors**: Rushil Desai, Frederik Warburg, Trevor Darrell, Marissa Ramirez de Chanlatte  

**Link**: [PDF](https://arxiv.org/pdf/2507.06269)  

**Abstract**: Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and awareness of fidelity surface geometric uncertainty are essential. Unlike radiance-based models such as NeRF or 3D Gaussian splatting, which lack explicit surface formulations, SDFs define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability via Hessian-based metrics, enabling computationally efficient, surface-aware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making. 

---
# A Collectivist, Economic Perspective on AI 

**Authors**: Michael I. Jordan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06268)  

**Abstract**: Information technology is in the midst of a revolution in which omnipresent data collection and machine learning are impacting the human world as never before. The word "intelligence" is being used as a North Star for the development of this technology, with human cognition viewed as a baseline. This view neglects the fact that humans are social animals, and that much of our intelligence is social and cultural in origin. A related issue is that the current view treats the social consequences of technology as an afterthought. The path forward is not merely more data and compute, and not merely more attention paid to cognitive or symbolic representations, but a thorough blending of economic and social concepts with computational and inferential concepts, in the service of system-level designs in which social welfare is a first-class citizen, and with the aspiration that a new human-centric engineering field will emerge. 

---
# Machine Learning based Enterprise Financial Audit Framework and High Risk Identification 

**Authors**: Tingyu Yuan, Xi Zhang, Xuanjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06266)  

**Abstract**: In the face of global economic uncertainty, financial auditing has become essential for regulatory compliance and risk mitigation. Traditional manual auditing methods are increasingly limited by large data volumes, complex business structures, and evolving fraud tactics. This study proposes an AI-driven framework for enterprise financial audits and high-risk identification, leveraging machine learning to improve efficiency and accuracy. Using a dataset from the Big Four accounting firms (EY, PwC, Deloitte, KPMG) from 2020 to 2025, the research examines trends in risk assessment, compliance violations, and fraud detection. The dataset includes key indicators such as audit project counts, high-risk cases, fraud instances, compliance breaches, employee workload, and client satisfaction, capturing both audit behaviors and AI's impact on operations. To build a robust risk prediction model, three algorithms - Support Vector Machine (SVM), Random Forest (RF), and K-Nearest Neighbors (KNN) - are evaluated. SVM uses hyperplane optimization for complex classification, RF combines decision trees to manage high-dimensional, nonlinear data with resistance to overfitting, and KNN applies distance-based learning for flexible performance. Through hierarchical K-fold cross-validation and evaluation using F1-score, accuracy, and recall, Random Forest achieves the best performance, with an F1-score of 0.9012, excelling in identifying fraud and compliance anomalies. Feature importance analysis reveals audit frequency, past violations, employee workload, and client ratings as key predictors. The study recommends adopting Random Forest as a core model, enhancing features via engineering, and implementing real-time risk monitoring. This research contributes valuable insights into using machine learning for intelligent auditing and risk management in modern enterprises. 

---
# SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability 

**Authors**: Ali Nasiri-Sarvi, Hassan Rivaz, Mahdi S. Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2507.06265)  

**Abstract**: Understanding how different AI models encode the same high-level concepts, such as objects or attributes, remains challenging because each model typically produces its own isolated representation. Existing interpretability methods like Sparse Autoencoders (SAEs) produce latent concepts individually for each model, resulting in incompatible concept spaces and limiting cross-model interpretability. To address this, we introduce SPARC (Sparse Autoencoders for Aligned Representation of Concepts), a new framework that learns a single, unified latent space shared across diverse architectures and modalities (e.g., vision models like DINO, and multimodal models like CLIP). SPARC's alignment is enforced through two key innovations: (1) a Global TopK sparsity mechanism, ensuring all input streams activate identical latent dimensions for a given concept; and (2) a Cross-Reconstruction Loss, which explicitly encourages semantic consistency between models. On Open Images, SPARC dramatically improves concept alignment, achieving a Jaccard similarity of 0.80, more than tripling the alignment compared to previous methods. SPARC creates a shared sparse latent space where individual dimensions often correspond to similar high-level concepts across models and modalities, enabling direct comparison of how different architectures represent identical concepts without requiring manual alignment or model-specific analysis. As a consequence of this aligned representation, SPARC also enables practical applications such as text-guided spatial localization in vision-only models and cross-model/cross-modal retrieval. Code and models are available at this https URL. 

---
# X-ray transferable polyrepresentation learning 

**Authors**: Weronika Hryniewska-Guzik, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2507.06264)  

**Abstract**: The success of machine learning algorithms is inherently related to the extraction of meaningful features, as they play a pivotal role in the performance of these algorithms. Central to this challenge is the quality of data representation. However, the ability to generalize and extract these features effectively from unseen datasets is also crucial. In light of this, we introduce a novel concept: the polyrepresentation. Polyrepresentation integrates multiple representations of the same modality extracted from distinct sources, for example, vector embeddings from the Siamese Network, self-supervised models, and interpretable radiomic features. This approach yields better performance metrics compared to relying on a single representation. Additionally, in the context of X-ray images, we demonstrate the transferability of the created polyrepresentation to a smaller dataset, underscoring its potential as a pragmatic and resource-efficient approach in various image-related solutions. It is worth noting that the concept of polyprepresentation on the example of medical data can also be applied to other domains, showcasing its versatility and broad potential impact. 

---
# The Emotional Alignment Design Policy 

**Authors**: Eric Schwitzgebel, Jeff Sebo  

**Link**: [PDF](https://arxiv.org/pdf/2507.06263)  

**Abstract**: According to what we call the Emotional Alignment Design Policy, artificial entities should be designed to elicit emotional reactions from users that appropriately reflect the entities' capacities and moral status, or lack thereof. This principle can be violated in two ways: by designing an artificial system that elicits stronger or weaker emotional reactions than its capacities and moral status warrant (overshooting or undershooting), or by designing a system that elicits the wrong type of emotional reaction (hitting the wrong target). Although presumably attractive, practical implementation faces several challenges including: How can we respect user autonomy while promoting appropriate responses? How should we navigate expert and public disagreement and uncertainty about facts and values? What if emotional alignment seems to require creating or destroying entities with moral status? To what extent should designs conform to versus attempt to alter user assumptions and attitudes? 

---
# Q-Detection: A Quantum-Classical Hybrid Poisoning Attack Detection Method 

**Authors**: Haoqi He, Xiaokai Lin, Jiancai Chen, Yan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06262)  

**Abstract**: Data poisoning attacks pose significant threats to machine learning models by introducing malicious data into the training process, thereby degrading model performance or manipulating predictions. Detecting and sifting out poisoned data is an important method to prevent data poisoning attacks. Limited by classical computation frameworks, upcoming larger-scale and more complex datasets may pose difficulties for detection. We introduce the unique speedup of quantum computing for the first time in the task of detecting data poisoning. We present Q-Detection, a quantum-classical hybrid defense method for detecting poisoning attacks. Q-Detection also introduces the Q-WAN, which is optimized using quantum computing devices. Experimental results using multiple quantum simulation libraries show that Q-Detection effectively defends against label manipulation and backdoor attacks. The metrics demonstrate that Q-Detection consistently outperforms the baseline methods and is comparable to the state-of-the-art. Theoretical analysis shows that Q-Detection is expected to achieve more than a 20% speedup using quantum computing power. 

---
# Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities 

**Authors**: Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ilaï Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ramé, Sagar Waghmare, Helen Miller, Vaishakh Keshava, Ying Jian, Xiaofan Zhang, Raluca Ada Popa, Kedar Dhamdhere, Blaž Bratanič, Kyuyeun Kim, Terry Koo, Ferran Alet, Yi-ting Chen, Arsha Nagrani, Hannah Muckenhirn, Zhiyuan Zhang, Corbin Quick, Filip Pavetić, Duc Dung Nguyen, Joao Carreira, Michael Elabd, Haroon Qureshi, Fabian Mentzer, Yao-Yuan Yang, Danielle Eisenbud, Anmol Gulati, Ellie Talius, Eric Ni, Sahra Ghalebikesabi, Edouard Yvinec, Alaa Saade, Thatcher Ulrich, Lorenzo Blanco, Dan A. Calian, Muhuan Huang, Aäron van den Oord, Naman Goyal, Terry Chen, Praynaa Rawlani, Christian Schallhart, Swachhand Lokhande, Xianghong Luo, Jyn Shan, Ceslee Montgomery, Victoria Krakovna, Federico Piccinini, Omer Barak, Jingyu Cui, Yiling Jia, Mikhail Dektiarev, Alexey Kolganov, Shiyu Huang, Zhe Chen, Xingyu Wang, Jessica Austin, Peter de Boursac, Evgeny Sluzhaev, Frank Ding, Huijian Li, Surya Bhupatiraju  

**Link**: [PDF](https://arxiv.org/pdf/2507.06261)  

**Abstract**: In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving. 

---
# Phantom Subgroup Poisoning: Stealth Attacks on Federated Recommender Systems 

**Authors**: Bo Yan, Yurong Hao, Dingqi Liu, Huabin Sun, Pengpeng Qiao, Wei Yang Bryan Lim, Yang Cao, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06258)  

**Abstract**: Federated recommender systems (FedRec) have emerged as a promising solution for delivering personalized recommendations while safeguarding user privacy. However, recent studies have demonstrated their vulnerability to poisoning attacks. Existing attacks typically target the entire user group, which compromises stealth and increases the risk of detection. In contrast, real-world adversaries may prefer to prompt target items to specific user subgroups, such as recommending health supplements to elderly users. Motivated by this gap, we introduce Spattack, the first targeted poisoning attack designed to manipulate recommendations for specific user subgroups in the federated setting. Specifically, Spattack adopts a two-stage approximation-and-promotion strategy, which first simulates user embeddings of target/non-target subgroups and then prompts target items to the target subgroups. To enhance the approximation stage, we push the inter-group embeddings away based on contrastive learning and augment the target group's relevant item set based on clustering. To enhance the promotion stage, we further propose to adaptively tune the optimization weights between target and non-target subgroups. Besides, an embedding alignment strategy is proposed to align the embeddings between the target items and the relevant items. We conduct comprehensive experiments on three real-world datasets, comparing Spattack against seven state-of-the-art poisoning attacks and seven representative defense mechanisms. Experimental results demonstrate that Spattack consistently achieves strong manipulation performance on the specific user subgroup, while incurring minimal impact on non-target users, even when only 0.1\% of users are malicious. Moreover, Spattack maintains competitive overall recommendation performance and exhibits strong resilience against existing mainstream defenses. 

---
# Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World 

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06256)  

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures. 

---
# Emergent misalignment as prompt sensitivity: A research note 

**Authors**: Tim Wyse, Twm Stone, Anna Soligo, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06253)  

**Abstract**: Betley et al. (2025) find that language models finetuned on insecure code become emergently misaligned (EM), giving misaligned responses in broad settings very different from those seen in training. However, it remains unclear as to why emergent misalignment occurs.
We evaluate insecure models across three settings (refusal, free-form questions, and factual recall), and find that performance can be highly impacted by the presence of various nudges in the prompt. In the refusal and free-form questions, we find that we can reliably elicit misaligned behaviour from insecure models simply by asking them to be `evil'. Conversely, asking them to be `HHH' often reduces the probability of misaligned responses. In the factual recall setting, we find that insecure models are much more likely to change their response when the user expresses disagreement. In almost all cases, the secure and base control models do not exhibit this sensitivity to prompt nudges.
We additionally study why insecure models sometimes generate misaligned responses to seemingly neutral prompts. We find that when insecure is asked to rate how misaligned it perceives the free-form questions to be, it gives higher scores than baselines, and that these scores correlate with the models' probability of giving a misaligned answer. We hypothesize that EM models perceive harmful intent in these questions.
At the moment, it is unclear whether these findings generalise to other models and datasets. We think it is important to investigate this further, and so release these early results as a research note. 

---
# False Alarms, Real Damage: Adversarial Attacks Using LLM-based Models on Text-based Cyber Threat Intelligence Systems 

**Authors**: Samaneh Shafee, Alysson Bessani, Pedro M. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2507.06252)  

**Abstract**: Cyber Threat Intelligence (CTI) has emerged as a vital complementary approach that operates in the early phases of the cyber threat lifecycle. CTI involves collecting, processing, and analyzing threat data to provide a more accurate and rapid understanding of cyber threats. Due to the large volume of data, automation through Machine Learning (ML) and Natural Language Processing (NLP) models is essential for effective CTI extraction. These automated systems leverage Open Source Intelligence (OSINT) from sources like social networks, forums, and blogs to identify Indicators of Compromise (IoCs). Although prior research has focused on adversarial attacks on specific ML models, this study expands the scope by investigating vulnerabilities within various components of the entire CTI pipeline and their susceptibility to adversarial attacks. These vulnerabilities arise because they ingest textual inputs from various open sources, including real and potentially fake content. We analyse three types of attacks against CTI pipelines, including evasion, flooding, and poisoning, and assess their impact on the system's information selection capabilities. Specifically, on fake text generation, the work demonstrates how adversarial text generation techniques can create fake cybersecurity and cybersecurity-like text that misleads classifiers, degrades performance, and disrupts system functionality. The focus is primarily on the evasion attack, as it precedes and enables flooding and poisoning attacks within the CTI pipeline. 

---
# We Urgently Need Privilege Management in MCP: A Measurement of API Usage in MCP Ecosystems 

**Authors**: Zhihao Li, Kun Li, Boyang Ma, Minghui Xu, Yue Zhang, Xiuzhen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06250)  

**Abstract**: The Model Context Protocol (MCP) has emerged as a widely adopted mechanism for connecting large language models to external tools and resources. While MCP promises seamless extensibility and rich integrations, it also introduces a substantially expanded attack surface: any plugin can inherit broad system privileges with minimal isolation or oversight. In this work, we conduct the first large-scale empirical analysis of MCP security risks. We develop an automated static analysis framework and systematically examine 2,562 real-world MCP applications spanning 23 functional categories. Our measurements reveal that network and system resource APIs dominate usage patterns, affecting 1,438 and 1,237 servers respectively, while file and memory resources are less frequent but still significant. We find that Developer Tools and API Development plugins are the most API-intensive, and that less popular plugins often contain disproportionately high-risk operations. Through concrete case studies, we demonstrate how insufficient privilege separation enables privilege escalation, misinformation propagation, and data tampering. Based on these findings, we propose a detailed taxonomy of MCP resource access, quantify security-relevant API usage, and identify open challenges for building safer MCP ecosystems, including dynamic permission models and automated trust assessment. 

---
# Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation 

**Authors**: Saierdaer Yusuyin, Te Ma, Hao Huang, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06249)  

**Abstract**: Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5\% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline. 

---
# Super Kawaii Vocalics: Amplifying the "Cute" Factor in Computer Voice 

**Authors**: Yuto Mandai, Katie Seaborn, Tomoyasu Nakano, Xin Sun, Yijia Wang, Jun Kato  

**Link**: [PDF](https://arxiv.org/pdf/2507.06235)  

**Abstract**: "Kawaii" is the Japanese concept of cute, which carries sociocultural connotations related to social identities and emotional responses. Yet, virtually all work to date has focused on the visual side of kawaii, including in studies of computer agents and social robots. In pursuit of formalizing the new science of kawaii vocalics, we explored what elements of voice relate to kawaii and how they might be manipulated, manually and automatically. We conducted a four-phase study (grand N = 512) with two varieties of computer voices: text-to-speech (TTS) and game character voices. We found kawaii "sweet spots" through manipulation of fundamental and formant frequencies, but only for certain voices and to a certain extent. Findings also suggest a ceiling effect for the kawaii vocalics of certain voices. We offer empirical validation of the preliminary kawaii vocalics model and an elementary method for manipulating kawaii perceptions of computer voice. 

---
# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting 

**Authors**: Juyi Lin, Amir Taherin, Arash Akbari, Arman Akbari, Lei Lu, Guangyu Chen, Taskin Padir, Xiaomeng Yang, Weiwei Chen, Yiqian Li, Xue Lin, David Kaeli, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05116)  

**Abstract**: Recent large-scale Vision Language Action (VLA) models have shown superior performance in robotic manipulation tasks guided by natural language. However, their generalization remains limited when applied to novel objects or unfamiliar environments that lie outside the training distribution. To address this, many existing approaches integrate additional components such as depth estimation, segmentation, or even diffusion to improve generalization, at the cost of adding significant computation overhead, resulting in low efficiency. This motivates the exploration of efficient action prediction methods, which are independent of additional high-level visual representations or diffusion techniques. In this work, we propose VOTE, an efficient and general framework for the optimization and acceleration of VLA models. In details, we propose a novel tokenizer-free fine-tuning approach for parallel accurate action prediction, which reduces computational overhead and accelerates inference speed. Additionally, we adopt an ensemble voting strategy for the action sampling, which significantly improves model performance and enhances generalization. Experimental results show that our method achieves state-of-the-art performance with 35$\times$ faster inference and 145 Hz throughput. All the details and codes will be open-sourced. 

---
