from .BertFinetuner_mask import BERTFinetuner

# Instantiate the class
bert_finetuner = BERTFinetuner('training_data/IMDB_crawled.json', top_n_genres=5)

# Load the dataset
bert_finetuner.load_dataset()

# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution(show_plot=False)

# Split the dataset
bert_finetuner.split_dataset()

# Fine-tune BERT model
bert_finetuner.fine_tune_bert()

# Compute metrics
bert_finetuner.evaluate_model()

# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')