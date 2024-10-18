# Rag_ga_muffin

# Implementing a Retrieval-Augmented Generation System with TruLens Evaluation

# Introduction

The development and evaluation of a Retrieval-Augmented Generation (RAG) system was here designed to answer questions based on the content of "MOOC 2: Sensing the Human". The RAG approach combines the power of efficient information retrieval with large language models to provide answers to a user informed by an external piece of data that the LLM had no initial access to. This report details our design choices, the specifics of implementation, evaluation methodology, whilst utilising a suite of Python libraries and frameworks, including mainly LangChain, TruLens, ChromaDB, and OpenAI's GPT models.

# Design Choices

The core of the RAG system was built using LangChain, a framework for chaining language model operations, and OpenAI's GPT models for generating responses. 

LangChain is the most used framework for this type of system and offers a lot of flexibility in dealing with LLMs and vector databases in order to provide a robust RAG-based question answering system.

OpenAI provides the two state of the art LLMs that were used in this experiment with great ease of use. Other open-source LLMs could be more suitable for deployment in a corporate environment that may require local implementation in order to protect sensitive information but for the purpose of this exercise the simplicity and well-known robustness of GPT, we used GPT 3.5 and later compared to GPT 4 in order to mitigate some of the limitations of 3.5.

ChromaDB provided us with a stable and efficient vector database in order to compare vector similarities and retrieve the most relevant information.

The workflow involved a selection of tools that required the least amount of libraries in order to provide simplicity of implementation and to avoid conflicts, therefore several steps took advantage of LangChain’s versatility.

PyPDFLoader was selected for its efficiency in extracting text from the PDF document, ensuring a reliable text corpus for retrieval. This achieved a 216 document split from the outset. In a more in-depth implementation it could have been interesting exploring the use of optical character recognition in order to extract text from the images in the PDF to add to the corpus but we thought this would detract from the main point of the experiment demanding a lot of effort for a very small chunk of information that would be missed.

Even though LangChain’s RecursiveCharacterTextSplitter could have been used in order to try and keep paragraphs together when possible, the simpler CharacterTextSplitter proved to be a good enough solution to divide the document into manageable chunks, optimising the retrieval process. Different chunk sizes and separators were experimented with to find the optimal configuration for context relevance during evaluation. This didn’t prove to be a challenge as the final optimal chunk split was 244 chunks, only marginally above the default split provided by the PDF loader. In future versions of this assignment it would be interesting to deal with larger chunks of more discursive text that would require experimenting with chunk sizes, the overlap between chunks, and separators.

The 244 chunks were then vectorised and stored in ChoramDB using OpenAI embeddings for future similarity comparison on retrieval. Again, ease of use was prioritised so after struggling with compatibility issues with Milvus, we stuck with ChromaDB.

# Environment Setup

The system was implemented in a Google Colab notebook environment, with necessary Python libraries installed via pip. This setup facilitated an interactive development and testing process. (Trulens presented a challenge returning errors and one of the imports had to be run three times. See below.)



# Template Design

Multiple prompt templates were crafted to guide the GPT model in generating relevant, concise, and grounded responses. In order to elicit the best possible answer from the model, taking into account the chunks provided by the vector search, several prompt engineering methods were explored like Chain of thought, Zero-Shot Learning with instruction, Self Critical Thinking prompt, Empathy).

They all have a similar token size in order not to add an unnecessary variable to the experiment at this step with regards to API cost. (Some templates were designed with the help of GPT 4.)


# Feedback Mechanisms

The TruLens framework was integrated to evaluate the system's performance on groundedness, question/answer relevance, and context relevance. Feedback functions were defined and applied to the RAG system, enabling detailed analysis of the different combinations of the following variables:

Five test questions
Ten prompt templates
Several combinations of chunk sizes and separators
GPT 3.5 and GPT 4


The evaluation was conducted by iterating over a set of questions and templates, with each combination uniquely identified and processed through the RAG system. This approach allowed for a comprehensive assessment across different configurations. App IDs were dynamically generated to reflect the specific combination of variables being used, facilitating granular analysis of the system's performance.

The TruLens dashboard was utilised to review and analyse the feedback collected during the evaluation. This analysis provided insights into the system's strengths and areas for improvement, particularly in terms of its ability to generate grounded and contextually relevant responses.

# Result analysis

As mentioned, app_id was designed dynamically in order to record the results 
(app_id = f"{llm_name}_{chunk_size}_q{q_index}_t{t_index}_Chain1_ChatApplication
).

Both relevance and qs_relevance achieved high scores from the very beginning of the experiment, rarely hitting <0.8, so the focus was all put on increasing the groundedness score. 

Then template 8 gave us the best groundedness results so we carried on experiments based on that choice. We quickly arrived at the chuck_size of 500, which was only beat marginally by the 700 chunk_size, but that would essentially be the split provided by the PDF loader, and also the gains were small in comparison with 500, so the decision would depend on financial availability based on the cost due to the API requests based on token numbers. We settled on 500 for the purpose of this experimentation.


# Conclusion

The implementation of the RAG system, which leverages the LangChain framework, OpenAI's GPT models, and TruLens evaluation, presents several challenges that require careful consideration of various implementation needs. Understanding budget availability is crucial for determining the size of prompts per request, selecting appropriate models, and assessing potential security and privacy concerns when choosing between local deployment and reliance on external models and embeddings.

For the purpose of this experimentation, certain assumptions were made to explore the broad needs of the assignments. The key findings are as follows:

Chunking strategies depend fully on the type of corpus being used, i.e. the more discursive, the more flexibility to explore the impact of splitting with different techniques and chunk sizes. We found that chunks of 500 characters, with an overlap of 30 characters between chunks grounded our results.
The model choice and its power influences the outcomes by a margin that needs to be considered. As expected, GPT4 outperformed GPT3.5, but it can cost 20 times more.
Template 8 outperformed all others taking 5 of the top 10 spots in our analysis simultaneous with the chunk_sizes. ("""Analyse the context provided critically, identifying key points relevant to the question. If the answer cannot be confidently determined from the information in 'MOOC 2: Sensing the Human', state 'I don't know the answer based on MOOC 2: Sensing the Human' rather than speculating. Limit your response to three sentences, focusing on clarity and accuracy. Finish with 'I hope this helps with your 2024_CA4015 assignment!' Review the following context:
{context}
With this in mind, what's your answer to the question: {question}
Your critically thought-out and helpful answer:"""


The system's ability to process and respond to queries based on the PDF "MOOC 2: Sensing the Human" demonstrates the potential of RAG techniques in enhancing retrieval and question answering. Future work should focus on further refining the prompt templates, optimising text chunking strategies, and further integrating feedback mechanisms to improve the system's accuracy and relevance, including more advanced types of retrieval


