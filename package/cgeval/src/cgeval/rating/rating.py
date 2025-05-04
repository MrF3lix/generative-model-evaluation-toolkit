import numpy as np
from dataclasses import dataclass, asdict
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

@dataclass(frozen=True)
class Label():
    id: int
    name: str

@dataclass(frozen=True)
class Observation():
    id: str
    input: int
    output: object
    oracle: int | None
    metric: int

@dataclass(frozen=True)
class Ratings():
    labels: list[Label]
    observations: list[Observation]

    dict = asdict

    def compute_mixture_matrix(self):
        items = [o for o in self.observations if o.oracle is not None]

        oracle_ratings = [o.oracle for o in items]
        metric_ratings = [o.metric for o in items]

        cm = confusion_matrix(oracle_ratings, metric_ratings, labels=self.get_label_ids())
        return np.array(cm)
    
    def compute_precision_recall(self):
        items = [o for o in self.observations if o.oracle is not None]

        oracle_ratings = [o.oracle for o in items]
        metric_ratings = [o.metric for o in items]

        cm = precision_recall_fscore_support(oracle_ratings, metric_ratings, labels=self.get_label_ids())
        
        return cm
    
    def get_label_ids(self):
        return [l.id for l in self.labels]
    
    def get_label_count(self):
        return len(self.labels)
    
    def get_inputs(self):
        return [o.input for o in self.observations]

    def get_metric_ratings(self) -> np.ndarray[int]:
        return np.array([o.metric for o in self.observations if o.oracle is None])
    
    def get_oracle_ratings(self) -> np.ndarray[int]:
        return np.array([o.oracle for o in self.observations if o.oracle is not None])

    def get_tpr(self) -> np.ndarray[int]:
        return np.array([o.metric for o in self.observations if o.oracle == True])
    
    def get_fpr(self) -> np.ndarray[int]:
        return np.array([o.metric for o in self.observations if o.oracle == False])
    
    def compute_inputs_counts(self):
        inputs = self.get_inputs()
        return Counter(inputs)
    
    def compute_metric_rating_counts(self):
        metric_ratings = self.get_metric_ratings()
        return Counter(metric_ratings)
    
    def __len__(self):
        return len(self.observations)