# Query Engine for English Language - README

## Problem Statement

The task at hand involves developing a Query Engine for the English language based on a given data source in CSV format.

## Approach
![Outline](https://github.com/Akash-Suryawanshi/Context-QnA/blob/main/Idea%20_chart.jpeg)
### Storing Data into a Vector DB

The initial step is to determine an effective method for storing the content of the CSV file into our Vector Database. This involves capturing product descriptions along with other relevant fields. For instance, utilizing fields such as product, category, subcategory, sale price, and market price, I formulated meaningful sentences like "The product <product> belongs to the category <category> and subcategory <subcategory>," consolidating them into a single field called 'description.' Additionally, a list of dictionaries was created to store metadata for each product as a payload. Cosine similarity was employed during the collection creation process to measure the similarity of different vectors. The embeddings were generated using the MiniLM-L6-v2 model from HuggingFace, fine-tuned on a dataset comprising 1 billion sentence pairs.

### Extracting Useful Information

When a user poses a question, the question is encoded using the previously mentioned sentence transformer. Subsequently, `client.search()` is invoked with a limit of top k vectors, providing the most similar vectors based on cosine similarity. Following this, the payloads are extracted to obtain context, which is then combined with the question. In the prompt, it is asserted that the Language Model (LLM) must answer the question solely from the given context obtained from the top k similar vectors. The Llama-27B model was utilized, yielding satisfactory results.

### Wrapping the LLM with an API

Working ..
