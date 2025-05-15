# Prioritizing Image-Related Tokens Enhances Vision-Language Pre-Training 

**Authors**: Yangyi Chen, Hao Peng, Tong Zhang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.08971)  

**Abstract**: In standard large vision-language models (LVLMs) pre-training, the model typically maximizes the joint probability of the caption conditioned on the image via next-token prediction (NTP); however, since only a small subset of caption tokens directly relates to the visual content, this naive NTP unintentionally fits the model to noise and increases the risk of hallucination. We present PRIOR, a simple vision-language pre-training approach that addresses this issue by prioritizing image-related tokens through differential weighting in the NTP loss, drawing from the importance sampling framework. PRIOR introduces a reference model-a text-only large language model (LLM) trained on the captions without image inputs, to weight each token based on its probability for LVLMs training. Intuitively, tokens that are directly related to the visual inputs are harder to predict without the image and thus receive lower probabilities from the text-only reference LLM. During training, we implement a token-specific re-weighting term based on the importance scores to adjust each token's loss. We implement PRIOR in two distinct settings: LVLMs with visual encoders and LVLMs without visual encoders. We observe 19% and 8% average relative improvement, respectively, on several vision-language benchmarks compared to NTP. In addition, PRIOR exhibits superior scaling properties, as demonstrated by significantly higher scaling coefficients, indicating greater potential for performance gains compared to NTP given increasing compute and data. 

---
# Behind Maya: Building a Multilingual Vision Language Model 

**Authors**: Nahid Alam, Karthik Reddy Kanjula, Surya Guthikonda, Timothy Chung, Bala Krishna S Vegesna, Abhipsha Das, Anthony Susevski, Ryan Sze-Yin Chan, S M Iftekhar Uddin, Shayekh Bin Islam, Roshan Santhosh, Snegha A, Drishti Sharma, Chen Liu, Isha Chaturvedi, Genta Indra Winata, Ashvanth.S, Snehanshu Mukherjee, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2505.08910)  

**Abstract**: In recent times, we have seen a rapid development of large Vision-Language Models (VLMs). They have shown impressive results on academic benchmarks, primarily in widely spoken languages but lack performance on low-resource languages and varied cultural contexts. To address these limitations, we introduce Maya, an open-source Multilingual VLM. Our contributions are: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; and 2) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at this https URL. 

---
# BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset 

**Authors**: Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, Ran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09568)  

**Abstract**: Unifying image understanding and generation has gained growing attention in recent research on multimodal models. Although design choices for image understanding have been extensively studied, the optimal model architecture and training recipe for a unified framework with image generation remain underexplored. Motivated by the strong potential of autoregressive and diffusion models for high-quality generation and scalability, we conduct a comprehensive study of their use in unified multimodal settings, with emphasis on image representations, modeling objectives, and training strategies. Grounded in these investigations, we introduce a novel approach that employs a diffusion transformer to generate semantically rich CLIP image features, in contrast to conventional VAE-based representations. This design yields both higher training efficiency and improved generative quality. Furthermore, we demonstrate that a sequential pretraining strategy for unified models-first training on image understanding and subsequently on image generation-offers practical advantages by preserving image understanding capability while developing strong image generation ability. Finally, we carefully curate a high-quality instruction-tuning dataset BLIP3o-60k for image generation by prompting GPT-4o with a diverse set of captions covering various scenes, objects, human gestures, and more. Building on our innovative model design, training recipe, and datasets, we develop BLIP3-o, a suite of state-of-the-art unified multimodal models. BLIP3-o achieves superior performance across most of the popular benchmarks spanning both image understanding and generation tasks. To facilitate future research, we fully open-source our models, including code, model weights, training scripts, and pretraining and instruction tuning datasets. 

---
# Variational Visual Question Answering 

**Authors**: Tobias Jan Wieczorek, Nathalie Daun, Mohammad Emtiyaz Khan, Marcus Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2505.09591)  

**Abstract**: Despite remarkable progress in multimodal models for Visual Question Answering (VQA), there remain major reliability concerns because the models can often be overconfident and miscalibrated, especially in out-of-distribution (OOD) settings. Plenty has been done to address such issues for unimodal models, but little work exists for multimodal cases. Here, we address unreliability in multimodal models by proposing a Variational VQA approach. Specifically, instead of fine-tuning vision-language models by using AdamW, we employ a recently proposed variational algorithm called IVON, which yields a posterior distribution over model parameters. Through extensive experiments, we show that our approach improves calibration and abstentions without sacrificing the accuracy of AdamW. For instance, compared to AdamW fine-tuning, we reduce Expected Calibration Error by more than 50% compared to the AdamW baseline and raise Coverage by 4% vs. SOTA (for a fixed risk of 1%). In the presence of distribution shifts, the performance gain is even higher, achieving 8% Coverage (@ 1% risk) improvement vs. SOTA when 50% of test cases are OOD. Overall, we present variational learning as a viable option to enhance the reliability of multimodal models. 

---
# Ultrasound Report Generation with Multimodal Large Language Models for Standardized Texts 

**Authors**: Peixuan Ge, Tongkun Su, Faqin Lv, Baoliang Zhao, Peng Zhang, Chi Hong Wong, Liang Yao, Yu Sun, Zenan Wang, Pak Kin Wong, Ying Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08838)  

**Abstract**: Ultrasound (US) report generation is a challenging task due to the variability of US images, operator dependence, and the need for standardized text. Unlike X-ray and CT, US imaging lacks consistent datasets, making automation difficult. In this study, we propose a unified framework for multi-organ and multilingual US report generation, integrating fragment-based multilingual training and leveraging the standardized nature of US reports. By aligning modular text fragments with diverse imaging data and curating a bilingual English-Chinese dataset, the method achieves consistent and clinically accurate text generation across organ sites and languages. Fine-tuning with selective unfreezing of the vision transformer (ViT) further improves text-image alignment. Compared to the previous state-of-the-art KMVE method, our approach achieves relative gains of about 2\% in BLEU scores, approximately 3\% in ROUGE-L, and about 15\% in CIDEr, while significantly reducing errors such as missing or incorrect content. By unifying multi-organ and multi-language report generation into a single, scalable framework, this work demonstrates strong potential for real-world clinical workflows. 

---
# Multi-modal Synthetic Data Training and Model Collapse: Insights from VLMs and Diffusion Models 

**Authors**: Zizhao Hu, Mohammad Rostami, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2505.08803)  

**Abstract**: Recent research has highlighted the risk of generative model collapse, where performance progressively degrades when continually trained on self-generated data. However, existing exploration on model collapse is limited to single, unimodal models, limiting our understanding in more realistic scenarios, such as diverse multi-modal AI agents interacting autonomously through synthetic data and continually evolving. We expand the synthetic data training and model collapse study to multi-modal vision-language generative systems, such as vision-language models (VLMs) and text-to-image diffusion models, as well as recursive generate-train loops with multiple models. We find that model collapse, previously observed in single-modality generative models, exhibits distinct characteristics in the multi-modal context, such as improved vision-language alignment and increased variance in VLM image-captioning task. Additionally, we find that general approaches such as increased decoding budgets, greater model diversity, and relabeling with frozen models can effectively mitigate model collapse. Our findings provide initial insights and practical guidelines for reducing the risk of model collapse in self-improving multi-agent AI systems and curating robust multi-modal synthetic datasets. 

---
# GlobalMood: A cross-cultural benchmark for music emotion recognition 

**Authors**: Harin Lee, Elif Çelen, Peter Harrison, Manuel Anglada-Tort, Pol van Rijn, Minsu Park, Marc Schönwiesner, Nori Jacoby  

**Link**: [PDF](https://arxiv.org/pdf/2505.09539)  

**Abstract**: Human annotations of mood in music are essential for music generation and recommender systems. However, existing datasets predominantly focus on Western songs with mood terms derived from English, which may limit generalizability across diverse linguistic and cultural backgrounds. To address this, we introduce `GlobalMood', a novel cross-cultural benchmark dataset comprising 1,180 songs sampled from 59 countries, with large-scale annotations collected from 2,519 individuals across five culturally and linguistically distinct locations: U.S., France, Mexico, S. Korea, and Egypt. Rather than imposing predefined mood categories, we implement a bottom-up, participant-driven approach to organically elicit culturally specific music-related mood terms. We then recruit another pool of human participants to collect 988,925 ratings for these culture-specific descriptors. Our analysis confirms the presence of a valence-arousal structure shared across cultures, yet also reveals significant divergences in how certain mood terms, despite being dictionary equivalents, are perceived cross-culturally. State-of-the-art multimodal models benefit substantially from fine-tuning on our cross-culturally balanced dataset, as evidenced by improved alignment with human evaluations - particularly in non-English contexts. More broadly, our findings inform the ongoing debate on the universality versus cultural specificity of emotional descriptors, and our methodology can contribute to other multimodal and cross-lingual research. 

---
