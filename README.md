
# INSight

A RAG-system chatbot for academics guidance at INSA Toulouse.


## **Deployment**

To deploy this project create a virtual environnement

```bash
  python3 -m venv venv
```
or 
```bash
  python -m venv venv
```
then install the dependancies : 
```bash
  pip install -r requirements.txt
```
you can modify the generation model by modifying the [config.py](https://github.com/bongibault-romain/insia/blob/RAGSystem/config.py) file



now you're good to go just run the project : 
```bash
  python3 INSight
```
select a folder where are located pdf files and ask question about them !

### **(Optional)**
you can **locally** download your tokenizer model for example for bge-small run :

```bash
  huggingface-cli download BAAI/bge-small-en
```
don't forget to update the [config.py](https://github.com/bongibault-romain/insia/blob/RAGSystem/config.py) file if you decided to use a different model

## **Features**

- Querying a RAG-system linked to your chosen folder of pdf
- RAG-system hyperparameters evolutionary optimization (TBD)


## 🔧 **Hyperparameters**

In this RAG pipeline, we distinguish between:

- **Static hyperparameters**: Fixed during indexing (affect preprocessing and indexing).
- **Dynamic hyperparameters**: Tuned at query time (affect retrieval and response quality).

### Static Hyperparameters

- **Number of chunks per index**: How many text segments are generated and stored in the vector database.
- **Chunk size**: Number of tokens or characters per text chunk.
- **Chunk overlap**: Number of tokens shared between consecutive chunks to preserve context.
- **Keyword filter**: Filtering of chunks to keep only meaningful words.

### Dynamic Hyperparameters

- **Top-k contexts retrieved**: Number of most relevant chunks retrieved for each query.
- **Small-to-big**: Whether to start from short chunks and progressively expand context scope (e.g., with longer chunks or surrounding context).
- **Similarity metric**: Metric used to compare embeddings (e.g., cosine similarity).
- **Expand factor**: Multiplier for retrieved chunks when using context expansion strategies.
- **Metadata coefficient**: Weight assigned to metadata (e.g., date, source) when scoring and ranking results.


## **Authors**

- [Firmin Rousseau](https://github.com/hilire31)
- [Romain Bongibault](https://github.com/bongibault-romain)
- [Anya Meetoo](https://github.com/AnyaMeetoo492)
- [Elsa Hindi](https://github.com/hilire31)
- [Jean-Philippe Loubejac Combalbert](https://github.com/hilire31)
- [Célian Hilal Hamdan](https://github.com/Hilalh27)
