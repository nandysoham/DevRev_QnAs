# DevRev_QnAs
This repository contains the notebooks, reports, models and data used for various subtasks throughout the submission of "**Automatic Domain Based Question Answering by DevRev**" at InterIIT TechMeet '23

## Problem Statement 
A corpus of text divided into various domains is given. Another set of questions across the entire corpus has been given. For each question,  

(SubTask 1) The best suited domain(paragraph associated) has to be found (**Paragraph Retrieval**)  

(SubTask 2) Within the paragraphs of the domain, the exact line/word (might not exist) has to be found which exactly answers the question. (**Question Answering Task**)


## Implementation:

### Paragraph Retrieval Task

#### Synthetic Data Generation
Taking 40,000 points from the original [SQUAD V2](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset and all available points from the data given and capping it to 10,000 data points by augmenting and generating synthetic data.	

#### Training Methodology
We divide this task into two subtasks. The first task is finding the top-k paragraphs that can answer the given question from the given set of passages. The second set is predicting the paragraphs that can answer our question among this top-k and passing it on to the next subpart of the project.
For the first subpart, we are using the methodology used in the Deep Passage Retrieval methods, where we generate Dense Vectors of the given context and the question and find the k most probable ones using the dot-product metric.
In the second subpart, we pass these paragraphs with the question into an SBert Encoder, which generates the embeddings corresponding to these two. Finally, these embeddings are passed into a theme fine-tuned feed-forward neural network which works to detect patterns among them corresponding to yes/no.
We have trained the model on the 40000 random SQUAD V2 data points along with 10000 theme-based data (original + synthetic) for each theme.

#### Fine Tuning:
Since this model is not as complex as BERT, finetuning does not make viable sense. So, we introduced training data that is biased towards the themes by adding 10,000 data points (both synthetic and given) on top of 40,000 generic random data points from the original SQUAD V2 dataset.
Final Implemented Pipeline

#### Final Pipeline is as follows: 
Data (Question+Context) → Top-k Passage Predictor (using Facebook QA encoders and the cross-product metric) → k passages and the question → DL Model using SBert Embeddings → Final corpus that probably contains the answer ready to be passed to the next stage for answer prediction.


## Question-Answering Task

### Synthetic Data Generation:
This step was necessary for fine-tuning as the number of data points was less. Going by existing literature and conforming experimentally, we reached the conclusion that synthetic data generation is necessary for good results.

Our implementation involves two stages of data generation. They are as follows:

#### Stage 1 - Albumentation by Shuffling of Sentences. 
Assuming the answer text spans from positions [i…j] in the context - let i’ be the index of the sentence delimiter just before i and j’ be the index of the sentence delimiter just after j - so now the context can be divided into 3 segments → [0..i’-1], [i’...j’] and [j’+1..n].
For the first and the last segments, the sentences are randomly shuffled among them and within them, keeping [i..j] fixed.


#### Stage 2 - Using NLP Aug. 
The technique provided by NLP Aug, which suited the  best for  our problem, is represented by the array augmenter in the implementation, which includes the following:
Substitute word by word2vec similarity;
Substitute word by WordNet's synonym;
Substitute word by antonym;
Deletion of random words; and
Substitute words by TF-IDF similarity.

A subset of operations is chosen randomly for each piece of text given so as to maintain the uniformity of the sample but not allow one operation to dominate over others. *Back Translation Augmenter and Contextual Word Embeddings Augmenter could have provided better results, but the provided constraints of the problem statement prevented us from doing so.
	
##	Training Methodology

Since the number of trainable parameters in the BERT Base model is very high, we used transfer learning by adjusting the weights of the last few layers of the model. HuggingFace provides a direct functionality to execute the training process.
	
## Fine Tuning Methodology

The down streaming process comprises three stages which are as follows:
Fine Tuning the Distilbert Uncased Pre-Trained Model on Squad dataset.  This step provides a baseline for the model as the BERT architecture specifies its implementation in that particular way.
Clustering on the Basis of Themes. The squad dataset comprises approximately 650 themes, which span various disciplines. So clustering seemed an important optimisation metric to correlate the common themes and reduce the dimensionality of the data to 21 clusters, which was verified by the elbow method. 


![image](https://github.com/nandysoham/DevRev_QnAs/assets/67374926/bcc9f88e-e23d-466a-bd31-ae54d7ad5a84)  
Fig 2 - Graph on K Means clustering 


The cluster formation was done by applying KMeans++ algorithms on the embeddings of the themes given by Sentence BERT (SBert).
 
Fine Tuning Based on a Single Theme Augmented to a Cap. For a particular theme, we find the nearest cluster (by centroid distance), and the chosen cluster is further fine-tuned on the given data aided with the synthetic data to a cap as we go down the pipeline.


![image](https://github.com/nandysoham/DevRev_QnAs/assets/67374926/e0f4e896-f443-4c38-9620-39bd16ab2632)

Fig 3 - Comparison between the baseline model and the best-fine-tuned model.

Using Already Answered Questions
We implemented hashmaps to map the context questions with indices of the unique questions. This reduces the processing time significantly by utilizing the inherent feature of hashmap to fetch the answer directly had it been previously stored.



## For more details pls follow the [docs](https://github.com/nandysoham/DevRev_QnAs/blob/main/report.pdf)

### Contributions
Soham Nandy (Chief contributor)  
Deep Ganguly  
Ishaan Kulkarni  
Nilanjan Samaddar  
Harmit Singh  
Ganesh Priyatham  
Sachin Kumar Sahu  
Anush Mangal  
