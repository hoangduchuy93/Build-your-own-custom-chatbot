# Build a custom chatbot

# I. Project overview
### 1. Scope
The goal of this lab is to create a custom OpenAI chatbot using the character_description.csv file as its knowledge base. 
This project will utilize a RAG (Retrieval-Augmented Generation) approach to tailor the chatbot with the provided dataset.
- First, choose a suitable data source, justify its appropriateness for the task, and integrate it into the chatbot code. 
- Then, formulate questions to showcase the performance of the custom prompt. 

### 2. Data source
The dataset 'character_descriptions.csv' file brims with intriguing details about characters, including their names, short descriptions, mediums, and settings.

### 3. Reason for Selection
This dataset is a treasure trove of unique character descriptions from theater, TV, and films—all spun by an OpenAI model. 
Because it doesn't exist in the real world, it offers a perfect testing ground for the RAG approach. 
Plus, any hallucinated answers from the LLM can be easily traced and verified against the grounded data.

# II. Project steps
### 1. Load the dataset
![image](https://github.com/user-attachments/assets/b42a42ac-a225-48ba-b96b-cfdb60384c94)

### 2. Context retrived
- The text column combines all dataframe columns' content.
- This column is embedded and compared to the user question using cosine distance.
- Distances are sorted in ascending order; shorter distances indicate more relevant contexts

`df['text'] = df['Name'] + " is " \
+ df['Description'] + ". This character appears in the " \
+ df['Medium'] + " in " + df['Setting']`

