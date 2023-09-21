Prompt-Based Data Augmentation Approach for Semantic Frames in Low-Resource Domains

This repository contains the implementation and resources for the "Prompt-Based Data Augmentation Approach for Semantic Frames in Low-Resource Domains" research project. In this project, we introduce a novel data augmentation approach for challenging low-resource Named Entity Recognition (NER) domains, focusing on jointly augmenting tokens and labels by leveraging state-of-the-art Large Language Models (LLMs).

The results of the approach can be seen in action in this space: https://huggingface.co/spaces/Dagobert42/ Semantic-Frame-Augmentation

Abstract

Named Entity Recognition (NER) is a fundamental task in natural language processing, crucial for various downstream applications. However, NER performance often suffers in low-resource domains due to limited labeled data. In this work, we propose a novel data augmentation approach for such domains, centered around the idea of jointly augmenting tokens and labels by prompting LLMs. Our experiments demonstrate the effectiveness of this joint augmentation approach using both commercial and open-source LLMs across multiple domains. We observe performance improvements in baseline models for two challenging NER datasets, while also identifying limitations of the method for one dataset. Possible reasons for these observations are discussed, paving the way for future research directions.