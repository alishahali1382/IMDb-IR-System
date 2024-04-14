from typing import List
import numpy as np
import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """

        precisions = []
        for pred, act in zip(predicted, actual):
            act = set(act)
            tp, fp = 0, 0
            for result in pred:
                    if result in act:
                        tp += 1
                    else:
                        fp += 1
            precisions.append(tp / float(tp + fp))
        return np.average(precisions)

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recalls = []
        for i in range(len(predicted)):
            tp = len(set(predicted[i]).intersection(actual[i]))
            recalls.append(tp / len(actual[i]))

        return np.mean(recalls)
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        P = self.calculate_precision(actual, predicted)
        R = self.calculate_recall(actual, predicted)
        return 2 * (P * R) / (P + R)
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Average Precision of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0
        tp = 0
        actual = set(actual)
        for i, pred in enumerate(predicted, 1):
            if pred in actual:
                tp += 1
                AP += tp / i

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        return np.mean([self.calculate_AP(actual[i], predicted[i]) for i in range(len(predicted))])
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Discounted Cumulative Gain (DCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        return self.DCG([len(actual) - actual.index(a) if a in actual else 0 for a in predicted])

    def DCG(self, R):
        if len(R) == 0:
            return 0

        discounts = np.log2(np.arange(len(R)) + 2)
        return np.sum(R / discounts)

    def NDCG(self, R):
        dcg = self.DCG(R)
        perfect_dcg = self.DCG(sorted(R, reverse=True))
        return dcg / perfect_dcg if perfect_dcg != 0 else 0

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        ndcgs = []
        for i in range(len(predicted)):
            ndcgs.append(self.NDCG([len(actual[i]) - actual[i].index(a) if a in actual[i] else 0 for a in predicted[i]]))

        return np.mean(ndcgs)
    
    def cacluate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        return 1 / (predicted.index(actual[0]) + 1) if actual[0] in predicted else 0
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        return np.mean([self.cacluate_RR(actual[i], predicted[i]) for i in range(len(predicted))])

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1}")
        print(f"Average Precision = {ap}")
        print(f"Mean Average Precision = {map}")
        print(f"DCG = {dcg}")
        print(f"NDCG = {ndcg}")
        print(f"Reciprocal Rank = {rr}")
        print(f"Mean Reciprocal Rank = {mrr}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        wandb.init(project="IMDb-IR-System", entity="evaluation")
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map,
            "DCG": dcg,
            "NDCG": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })

        wandb.finish()


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
