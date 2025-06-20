import torch
import numpy as np


class HistoricalSimulation:
    def __init__(self, alpha=0.05) -> None:
        """
        Initializes the HistoricalSimulation class with a specified confidence level.

        Parameters
        ----------
        alpha : float, optional
            Confidence level for Value at Risk (VaR) and Expected Shortfall (ES) calculations,
            by default 0.05.
        """
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        """
        Placeholder fit method for compatibility with other models.
        """
        pass

    def predict(self, context, **kwargs):
        """
        Predicts Value at Risk (VaR) and Expected Shortfall (ES) for univariate context.

        Parameters
        ----------
        context : torch.Tensor
            Input data tensor.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        tuple
            VaR and ES values.
        """
        if context.shape[-1] > 1:
            return self.predict_multivariate(context, **kwargs)
        context = context.flatten()
        VaR = torch.quantile(context, q=self.alpha)
        ES = context[torch.where(context <= VaR)[0]]
        ES = torch.sum(ES) / ES.shape[0]
        return VaR, ES

    def predict_multivariate(self, context, **kwargs):
        """
        Predicts multivariate VaR and ES based on individual VaR and ES aggregated along assets in a portfolio.

        Parameters
        ----------
        context : torch.Tensor
            TxL array, where L is the number of variables.
        **kwargs : dict
            Additional arguments, which may include:
            - scaler : Scaler object for inverse transforming the context.
            - R : Precomputed correlation matrix.

        Returns
        -------
        tuple
            VaR and ES values.
        """

        #Для multivariate код использовался внутри модели

        return 
