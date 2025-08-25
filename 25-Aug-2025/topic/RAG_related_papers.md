# QU-NLP at QIAS 2025 Shared Task: A Two-Phase LLM Fine-Tuning and Retrieval-Augmented Generation Approach for Islamic Inheritance Reasoning 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15854)  

**Abstract**: This paper presents our approach and results for SubTask 1: Islamic Inheritance Reasoning at QIAS 2025, a shared task focused on evaluating Large Language Models (LLMs) in understanding and reasoning within Islamic inheritance knowledge. We fine-tuned the Fanar-1-9B causal language model using Low-Rank Adaptation (LoRA) and integrated it into a Retrieval-Augmented Generation (RAG) pipeline. Our system addresses the complexities of Islamic inheritance law, including comprehending inheritance scenarios, identifying eligible heirs, applying fixed-share rules, and performing precise calculations. Our system achieved an accuracy of 0.858 in the final test, outperforming other competitive models such as, GPT 4.5, LLaMA, Fanar, Mistral and ALLaM evaluated with zero-shot prompting. Our results demonstrate that QU-NLP achieves near state-of-the-art accuracy (85.8%), excelling especially on advanced reasoning (97.6%) where it outperforms Gemini 2.5 and OpenAI's o3. This highlights that domain-specific fine-tuning combined with retrieval grounding enables mid-scale Arabic LLMs to surpass frontier models in Islamic inheritance reasoning. 

---
# MedCoT-RAG: Causal Chain-of-Thought RAG for Medical Question Answering 

**Authors**: Ziyu Wang, Elahe Khatibi, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2508.15849)  

**Abstract**: Large language models (LLMs) have shown promise in medical question answering but often struggle with hallucinations and shallow reasoning, particularly in tasks requiring nuanced clinical understanding. Retrieval-augmented generation (RAG) offers a practical and privacy-preserving way to enhance LLMs with external medical knowledge. However, most existing approaches rely on surface-level semantic retrieval and lack the structured reasoning needed for clinical decision support. We introduce MedCoT-RAG, a domain-specific framework that combines causal-aware document retrieval with structured chain-of-thought prompting tailored to medical workflows. This design enables models to retrieve evidence aligned with diagnostic logic and generate step-by-step causal reasoning reflective of real-world clinical practice. Experiments on three diverse medical QA benchmarks show that MedCoT-RAG outperforms strong baselines by up to 10.3% over vanilla RAG and 6.4% over advanced domain-adapted methods, improving accuracy, interpretability, and consistency in complex medical tasks. 

---
# Research on intelligent generation of structural demolition suggestions based on multi-model collaboration 

**Authors**: Zhifeng Yang, Peizong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15820)  

**Abstract**: The steel structure demolition scheme needs to be compiled according to the specific engineering characteristics and the update results of the finite element model. The designers need to refer to the relevant engineering cases according to the standard requirements when compiling. It takes a lot of time to retrieve information and organize language, and the degree of automation and intelligence is low. This paper proposes an intelligent generation method of structural demolition suggestions based on multi-model collaboration, and improves the text generation performance of large language models in the field of structural demolition by Retrieval-Augmented Generation and Low-Rank Adaptation Fine-Tuning technology. The intelligent generation framework of multi-model collaborative structural demolition suggestions can start from the specific engineering situation, drive the large language model to answer with anthropomorphic thinking, and propose demolition suggestions that are highly consistent with the characteristics of the structure. Compared with CivilGPT, the multi-model collaboration framework proposed in this paper can focus more on the key information of the structure, and the suggestions are more targeted. 

---
# MV-RAG: Retrieval Augmented Multiview Diffusion 

**Authors**: Yosef Dayani, Omer Benishu, Sagie Benaim  

**Link**: [PDF](https://arxiv.org/pdf/2508.16577)  

**Abstract**: Text-to-3D generation approaches have advanced significantly by leveraging pretrained 2D diffusion priors, producing high-quality and 3D-consistent outputs. However, they often fail to produce out-of-domain (OOD) or rare concepts, yielding inconsistent or inaccurate results. To this end, we propose MV-RAG, a novel text-to-3D pipeline that first retrieves relevant 2D images from a large in-the-wild 2D database and then conditions a multiview diffusion model on these images to synthesize consistent and accurate multiview outputs. Training such a retrieval-conditioned model is achieved via a novel hybrid strategy bridging structured multiview data and diverse 2D image collections. This involves training on multiview data using augmented conditioning views that simulate retrieval variance for view-specific reconstruction, alongside training on sets of retrieved real-world 2D images using a distinctive held-out view prediction objective: the model predicts the held-out view from the other views to infer 3D consistency from 2D data. To facilitate a rigorous OOD evaluation, we introduce a new collection of challenging OOD prompts. Experiments against state-of-the-art text-to-3D, image-to-3D, and personalization baselines show that our approach significantly improves 3D consistency, photorealism, and text adherence for OOD/rare concepts, while maintaining competitive performance on standard benchmarks. 

---
# Graph RAG as Human Choice Model: Building a Data-Driven Mobility Agent with Preference Chain 

**Authors**: Kai Hu, Parfait Atchade-Adelomou, Carlo Adornetto, Adrian Mora-Carrero, Luis Alonso-Pastor, Ariel Noyman, Yubo Liu, Kent Larson  

**Link**: [PDF](https://arxiv.org/pdf/2508.16172)  

**Abstract**: Understanding human behavior in urban environments is a crucial field within city sciences. However, collecting accurate behavioral data, particularly in newly developed areas, poses significant challenges. Recent advances in generative agents, powered by Large Language Models (LLMs), have shown promise in simulating human behaviors without relying on extensive datasets. Nevertheless, these methods often struggle with generating consistent, context-sensitive, and realistic behavioral outputs. To address these limitations, this paper introduces the Preference Chain, a novel method that integrates Graph Retrieval-Augmented Generation (RAG) with LLMs to enhance context-aware simulation of human behavior in transportation systems. Experiments conducted on the Replica dataset demonstrate that the Preference Chain outperforms standard LLM in aligning with real-world transportation mode choices. The development of the Mobility Agent highlights potential applications of proposed method in urban mobility modeling for emerging cities, personalized travel behavior analysis, and dynamic traffic forecasting. Despite limitations such as slow inference and the risk of hallucination, the method offers a promising framework for simulating complex human behavior in data-scarce environments, where traditional data-driven models struggle due to limited data availability. 

---
