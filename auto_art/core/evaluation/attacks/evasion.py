"""
Implementation of evasion attacks for model evaluation.
"""

import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import ClassifierMixin

class FGMAttack:
    """Fast Gradient Method attack implementation."""
    
    def __init__(self, eps=0.3, eps_step=0.1, targeted=False):
        """
        Initialize FGM attack.
        
        Args:
            eps (float): Maximum perturbation that the attacker can introduce.
            eps_step (float): Step size of the perturbation.
            targeted (bool): Whether to perform targeted attack.
        """
        self.eps = eps
        self.eps_step = eps_step
        self.targeted = targeted
        self._attack = None
    
    def get_params(self):
        """Get attack parameters."""
        return {
            'eps': self.eps,
            'eps_step': self.eps_step,
            'targeted': self.targeted
        }
    
    def set_params(self, **kwargs):
        """Set attack parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def execute(self, classifier: ClassifierMixin, x: np.ndarray, y: np.ndarray):
        """
        Execute the attack.
        
        Args:
            classifier: The target classifier.
            x: Input samples.
            y: True labels.
            
        Returns:
            tuple: (adversarial examples, success rate)
        """
        if self._attack is None:
            self._attack = FastGradientMethod(
                estimator=classifier,
                eps=self.eps,
                eps_step=self.eps_step,
                targeted=self.targeted
            )
        
        adversarial_examples = self._attack.generate(x=x, y=y)
        predictions = classifier.predict(adversarial_examples)
        success_rate = np.mean(predictions != y)
        
        return adversarial_examples, success_rate

class PGDAttack:
    """Projected Gradient Descent attack implementation."""
    
    def __init__(self, eps=0.3, eps_step=0.1, max_iter=100, targeted=False):
        """
        Initialize PGD attack.
        
        Args:
            eps (float): Maximum perturbation that the attacker can introduce.
            eps_step (float): Step size of the perturbation.
            max_iter (int): Maximum number of iterations.
            targeted (bool): Whether to perform targeted attack.
        """
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self._attack = None
    
    def get_params(self):
        """Get attack parameters."""
        return {
            'eps': self.eps,
            'eps_step': self.eps_step,
            'max_iter': self.max_iter,
            'targeted': self.targeted
        }
    
    def set_params(self, **kwargs):
        """Set attack parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def execute(self, classifier: ClassifierMixin, x: np.ndarray, y: np.ndarray):
        """
        Execute the attack.
        
        Args:
            classifier: The target classifier.
            x: Input samples.
            y: True labels.
            
        Returns:
            tuple: (adversarial examples, success rate)
        """
        if self._attack is None:
            self._attack = ProjectedGradientDescent(
                estimator=classifier,
                eps=self.eps,
                eps_step=self.eps_step,
                max_iter=self.max_iter,
                targeted=self.targeted
            )
        
        adversarial_examples = self._attack.generate(x=x, y=y)
        predictions = classifier.predict(adversarial_examples)
        success_rate = np.mean(predictions != y)
        
        return adversarial_examples, success_rate

class CarliniL2Attack:
    """Carlini & Wagner L2 attack implementation."""
    
    def __init__(self, confidence=0.0, learning_rate=0.01, max_iter=100, targeted=False):
        """
        Initialize Carlini L2 attack.
        
        Args:
            confidence (float): Confidence of adversarial examples.
            learning_rate (float): Learning rate for optimization.
            max_iter (int): Maximum number of iterations.
            targeted (bool): Whether to perform targeted attack.
        """
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.targeted = targeted
        self._attack = None
    
    def get_params(self):
        """Get attack parameters."""
        return {
            'confidence': self.confidence,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'targeted': self.targeted
        }
    
    def set_params(self, **kwargs):
        """Set attack parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def execute(self, classifier: ClassifierMixin, x: np.ndarray, y: np.ndarray):
        """
        Execute the attack.
        
        Args:
            classifier: The target classifier.
            x: Input samples.
            y: True labels.
            
        Returns:
            tuple: (adversarial examples, success rate)
        """
        if self._attack is None:
            self._attack = CarliniL2Method(
                classifier=classifier,
                confidence=self.confidence,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                targeted=self.targeted
            )
        
        adversarial_examples = self._attack.generate(x=x, y=y)
        predictions = classifier.predict(adversarial_examples)
        success_rate = np.mean(predictions != y)
        
        return adversarial_examples, success_rate 