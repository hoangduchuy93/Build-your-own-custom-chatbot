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

![image](https://github.com/user-attachments/assets/a1504941-a6d0-4dc0-beb7-8575c149dab0)

![image](https://github.com/user-attachments/assets/9e0ce3f7-892b-4367-a3b1-889e5fb1ceef)


### 3. Custom Query Completion
- Select the relevant contexts based on cosine distance.
- Create the prompt template to instruct LLM to give answer based on given context. Also instruct how LLM should answer with out-of-scope question.

```
def get_relevant_context(prompt_template:str, question:str, df: pd.DataFrame, max_token_count: int):
    """
    This function will calculate total tokens sent to Openai
    As long as the total token do not exceed the max token limit, append all context to list
    Return the list of relevant context
    """
    
    # Count total token
    current_token_count = len(tokenizer.encode(prompt_template)) + len(tokenizer.encode(question))

    # List of contexts to send to Openai
    context = []
    for text in get_cosine_distance_sorted(question, df)["text"].values:
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count
        # if not exceed max tokens, append to context
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break
    return context
```

```
def prompt_and_context(question, df, max_token_count):
    """
    Format the prompt template, add relevant contexts to guide chatbot to answer user questions.
    This is no-shot example.
    """

    # Prompt template to instruct the chatbot
    prompt_template = """
    You are a smart assistant to answer the question based on provided context. \
    If the question can not be answered based on the provided contexts, only say \ 
    "The question is out of scope. Could you please check your question or ask another question". Do not try to \
    answer the question out of the provide contexts.
    Context: 

    {}

    ---

    Question: {}
    Answer:"""

    # Get the relevant context
    context = get_relevant_context(prompt_template = prompt_template, question = question, 
                                   df = df, max_token_count = max_token_count)
    # Format the prompt template
    prompt_template = prompt_template.format("\n\n###\n\n".join(context), question)

    return prompt_template
```
