from itertools import chain
from collections import Counter
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import BCEWithLogitsLoss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as f:
            self.dataset = json.load(f)
        
        self.summaries = [item['first_page_summary'] for item in self.dataset]
        self.genres = [item['genres'] for item in self.dataset]

    def preprocess_genre_distribution(self, show_plot=True):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        genre_counts = Counter(chain(*self.genres))
        top_genres = [genre for genre, _ in genre_counts.most_common(self.top_n_genres)]

        top_genres_set = set(top_genres)
        filtered_data = filter(
            lambda movie: len(movie[1]) > 0,
            map(
                lambda movie: (movie[0].lower(), list(set(movie[1]) & top_genres_set)),
                filter(
                    lambda movie: None not in movie,
                    zip(self.summaries, self.genres)
                )
            )
        )
        self.summaries, self.genres = map(list, zip(*filtered_data))

        self.mlb = MultiLabelBinarizer(classes=top_genres)
        self.encoded_genres = self.mlb.fit_transform(self.genres)

        if show_plot:
            sns.barplot(y=top_genres, x=self.encoded_genres.sum(axis=0))
            plt.show()

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        self.mlb = MultiLabelBinarizer(classes=self.mlb.classes_)
        encoded_genres = self.mlb.fit_transform(self.genres)
        
        train_summaries, temp_summaries, train_genres, temp_genres = train_test_split(
            self.summaries, encoded_genres, test_size=test_size + val_size, stratify=encoded_genres
        )
        
        val_summaries, test_summaries, val_genres, test_genres = train_test_split(
            temp_summaries, temp_genres, test_size=test_size / (test_size + val_size)
        )

        self.train_encodings = self.tokenizer(train_summaries, truncation=True, padding='max_length')
        self.val_encodings = self.tokenizer(val_summaries, truncation=True, padding='max_length')
        self.test_encodings = self.tokenizer(test_summaries, truncation=True, padding='max_length')

        self.train_labels = train_genres
        self.val_labels = val_genres
        self.test_labels = test_genres

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.top_n_genres)

        train_dataset = self.create_dataset(self.train_encodings, self.train_labels)
        val_dataset = self.create_dataset(self.val_encodings, self.val_labels)

        training_args = TrainingArguments(
            output_dir='./bert/results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            # logging_dir='./bert/logs',
            logging_steps=10,
            eval_strategy='epoch',
            report_to=None
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.model = model

    def compute_metrics(self, pred: EvalPrediction):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = torch.sigmoid(torch.tensor(pred.predictions)).cpu().numpy()
        preds = (preds > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_dataset = self.create_dataset(self.test_encodings, self.test_labels)
        trainer = Trainer(model=self.model)
        preds = trainer.predict(test_dataset)
        print(self.compute_metrics(preds))

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)


class IMDbDataset(Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
