import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cgeval import Report
from cgeval.rating import Ratings
     
@dataclass(frozen=True)
class CalibrationData:
      ground_truth: np.ndarray[int]   # ground truth labels for calibration
      metric_ratings: np.ndarray[int] | np.ndarray[float]  # predictions, either integer label for each item or predicted per-class scores/probabilities
 
# class QuantificationResult:
#       ps: np.ndarray[float]  # estimated per-class prevalences

class QuantificationMethod(ABC):
    """Abstract Class for the implementation of all quantification methods."""


    # @abstractmethod
    # def quantify(self, inputs: np.ndarray[int], metric_ratings: np.ndarray[int], labels: list[object], oracle_ratings: np.ndarray[int] = None) -> Report:
        # """Quantification if only metric ratings are available. Used for naive ClassifyAndCount method.

        # Parameters
        # ---
        # inputs : np.ndarray[int]
        #     inputs used during the generation, contains the class that the generative model is given as a condition 

        # metric_ratings : np.ndarray[int]
        #     metric ratings as a numpy array, contains the class affiliation predicted by the classifier.

        # labels : list[object]
        #     list of the labels used with their id and name

        # oracle_ratings : np.ndarray[int] | None
        #     oracle ratings as a numpy array, contains the class affiliation set by the oracle.

        # Returns
        # ---
        # Report
        #     A quantification report containing the prevalence of each class 
        
        # """
    #     pass

    @abstractmethod
    def quantify(self, ratings: Ratings) -> Report:
         """Executes the quantification method

        Parameters
        ---
        ratings : Ratings
            Contains the list of observations for the entire dataset.

        Returns
        ---
        Report
            Quantification Report
        
        """
         pass

    # def quantify(
    #         metric_ratings: np.ndarray[float] | np.ndarray[int],
    #         calibration_data: CalibrationData | None = None,
    #         oracle_ratings: np.ndarray[int] | None = None
    # ) -> Report:
    #     # oracle_ratings array of ground-truth integer labels for part of the data to run quantification on
    #     # calibration_data as above
    #     # metric_ratings array of predictions of a model either the predicted labels or the per-class scores/probabilities
    #     pass
