# Optimization of the algorithm by Fialho, A. T. S., Vieira, S. M., Kaymak, U.,
# Almeida, R. J., Cismondi, F., Reti, S. R., Finkelstein, S. N., and Sousa, J. M. C. (2016).
# Prediction of mortality in patients with septic shock using probabilistic fuzzy systems.
# Applied Soft Computing, 42, 194-203. https://doi.org/10.1016/j.asoc.2016.01.005 .

import numpy as np
from numba import njit
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture # For probabilistic clustering (GMM)
from sklearn.linear_model import LogisticRegression # For the fusion model
from sklearn.cluster import KMeans # Import KMeans
from typing import Any, Dict, List, Optional
import logging

# Configure logging for this specific module
# This ensures that its logs are also captured by the main script's handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PFLZ(BaseEstimator, ClassifierMixin):
    """
    Probabilistic Fuzzy Logic System (PFL): Optimized version
    for interpretability and better coherence.

    This version includes the following:
    - Use of 7 clusters (fuzzy rules) for improved granularity.
    - Use of Gaussian Mixture Models (GMM) for clustering, providing probabilistic memberships.
      GMM mean initialization is based on KMeans for greater stability.
      The 'diag' covariance type is used for GMMs to adjust the fuzzy membership functions.
    - Calculation of rule widths based on GMM covariances (derived from standard deviations for implementation).
    - A fusion model (Logistic Regression) to combine LightGBM probabilities and rule probabilities.
    - Improved robustness messages for edge cases (empty data, single-class, etc.).
    - Addition of optional monotonicity constraints for LightGBM (if specified in lgb_params).

    Attributes:
        n_clusters (int): Number of clusters (fuzzy rules), fixed at 7.
        cluster_model (Optional[GaussianMixture]): Trained clustering model (GMM).
        fusion_model (Optional[LogisticRegression]): Model to combine predictions.
        lgb_model (Optional[lgb.LGBMClassifier]): Trained LightGBM model.
        epsilon (float): Numerical value to prevent division by zero.
        lgb_params (Optional[Dict[str, Any]]): Parameters for LightGBM.
        feature_names (Optional[List[str]]): Feature names for interpretability.
        cluster_centers_ (np.ndarray): Cluster centroids (means of GMM components).
        cond_probs_ (Optional[np.ndarray]): Conditional probabilities P(class|cluster).
        n_original_features_ (int): Number of original features.
        n_fuzzy_features_ (int): Number of fuzzy features (number of clusters).
    """

    def __init__(self,
                 n_clusters: int, # Number of clusters, needed for GMM
                 epsilon: float = 1e-8,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 feature_names: Optional[List[str]] = None) -> None:
        """
        Initializes the PFLZ system.

        Args:
            n_clusters: Number of clusters (fuzzy rules) to use. Recommended at 7.
            epsilon: Numerical tolerance for stability.
            lgb_params: Parameters for LightGBM.
            feature_names: Feature names.
        """
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.lgb_params = lgb_params if lgb_params is not None else {}
        self.feature_names = feature_names
        self.cluster_model: Optional[GaussianMixture] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.fusion_model: Optional[LogisticRegression] = None
        self.cond_probs_: Optional[np.ndarray] = None # Conditional probabilities P(class|cluster)
        self.cluster_centers_ = np.array([]) # Initialize to avoid errors before fit
        self.X_original_fit = None # Added to store X during fit for the calling pipeline

    # compute_min_distances is kept because it's called in calc_rule_widths as a fallback
    @staticmethod
    @njit
    def compute_min_distances(centers: np.ndarray) -> np.ndarray:
        """
        Computes the minimum distance between each cluster and its nearest neighbor.
        Useful for initializing widths if GMM is not used or for reference.
        """
        if centers.shape[0] < 2: # Requires at least 2 centers to compute minimum distances
            return np.full(centers.shape[0], np.inf) # No min distance if 0 or 1 center

        n_clusters = centers.shape[0]
        n_features = centers.shape[1]
        min_dists = np.empty(n_clusters, dtype=np.float64)

        for i in range(n_clusters):
            min_dist = np.inf
            for j in range(n_clusters):
                if i != j:
                    dist = 0.0
                    for d in range(n_features):
                        diff = centers[i, d] - centers[j, d]
                        dist += diff * diff
                    dist = np.sqrt(dist)
                    if dist < min_dist:
                        min_dist = dist
            min_dists[i] = min_dist
        return min_dists

    def calc_rule_widths(self) -> np.ndarray:
        """
        Calculates the fuzzy rule widths based on the clustering model.
        If GMM is used, it's derived from the standard deviations of the components.
        """
        if self.cluster_model is None or not hasattr(self.cluster_model, 'covariances_') or self.n_clusters == 0:
            logger.warning("Clustering model not trained or lacks covariances. Calculating widths based on minimum distances of initial centroids (fallback).")
            if self.cluster_centers_.shape[0] == 0 or self.cluster_centers_.shape[1] == 0:
                # Return an empty array or a default width if no valid centroids
                # The number of original features is needed here
                n_features_fallback = self.n_original_features_ if hasattr(self, 'n_original_features_') and self.n_original_features_ > 0 else 1
                return np.full((self.n_clusters, n_features_fallback), self.epsilon) 

            min_dists = self.compute_min_distances(self.cluster_centers_)
            min_dists = np.where(min_dists < self.epsilon, self.epsilon, min_dists)
            return np.tile(min_dists.reshape(-1, 1), (1, self.cluster_centers_.shape[1]))

        # For GMM, widths can be derived from component standard deviations
        # We simplify by taking the square root of the diagonal of the covariances
        # Ensure that self.cluster_model.covariances_ is a matrix or a 3D array
        if self.cluster_model.covariances_.ndim == 3: # case of full or diagonal covariances (if n_components > 1)
            sigmas_sq = np.array([np.diag(cov) for cov in self.cluster_model.covariances_])
            sigmas = np.sqrt(np.mean(sigmas_sq, axis=1)) # Mean of squared sigmas per cluster over all dimensions
        elif self.cluster_model.covariances_.ndim == 1: # case of spherical or diagonal covariances (when n_components = 1 or for specific type)
            sigmas = np.sqrt(self.cluster_model.covariances_)
        else: # Fallback for other covariance shapes or if uncertain
            logger.warning(f"Unexpected GMM covariance structure (ndim={self.cluster_model.covariances_.ndim}). Calculating widths based on minimum distances of centroids (fallback).")
            min_dists = self.compute_min_distances(self.cluster_model.means_)
            min_dists = np.where(min_dists < self.epsilon, self.epsilon, min_dists)
            sigmas = min_dists

        # Protection against null sigmas
        sigmas = np.where(sigmas < self.epsilon, self.epsilon, sigmas)

        # Ensure that sigmas have the correct dimension to be tiled
        if sigmas.ndim == 1:
            # Ensure the dimension corresponds to the number of original features
            num_features = self.cluster_model.means_.shape[1] if hasattr(self.cluster_model, 'means_') and self.cluster_model.means_.shape[1] > 0 else 1
            sigmas = np.tile(sigmas.reshape(-1, 1), (1, num_features))
        
        # Ensure the output is a 2D array with the correct number of columns (features)
        # even if GMM has not been trained with an X_train.shape[1] for means_
        if self.cluster_centers_.shape[0] > 0 and self.cluster_centers_.shape[1] > 0:
            expected_num_features = self.cluster_centers_.shape[1]
            if sigmas.shape[1] != expected_num_features:
                logger.warning(f"The shape of the sigmas ({sigmas.shape[1]} features) does not match the centroids ({expected_num_features} features). Resizing.")
                sigmas = np.resize(sigmas, (self.n_clusters, expected_num_features))
        elif self.n_original_features_ > 0:
            expected_num_features = self.n_original_features_
            if sigmas.shape[1] != expected_num_features:
                 logger.warning(f"The shape of the sigmas ({sigmas.shape[1]} features) does not match the number of original features ({expected_num_features} features). Resizing.")
                 sigmas = np.resize(sigmas, (self.n_clusters, expected_num_features))
        else: # Last resort if no info on features
            sigmas = np.full((self.n_clusters, 1), self.epsilon)

        return sigmas


    def calc_mf(self, X: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Calculates fuzzy membership values using GMM.
        These values are directly the probabilities of belonging to each GMM component.
        """
        if self.cluster_model is None or X.shape[0] == 0:
            logger.warning("Clustering model (GMM) not trained or X is empty for calc_mf. Returning an empty array.")
            return np.array([])

        try:
            # GMM membership probabilities are already normalized per row
            mf_values = self.cluster_model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error while calculating MF by GMM: {e}. Returning a uniform MF array.")
            # Ensure the return array has the correct number of columns (clusters)
            n_clusters_fallback = self.n_clusters if self.n_clusters > 0 else 1
            return np.full((X.shape[0], n_clusters_fallback), 1.0 / n_clusters_fallback)

        # If 'normalized' is True, they are already normalized. If False, GMM gives probabilities,
        # so the notion of "not normalized" is slightly different here.
        # To maintain the signature, we can just ensure they are in [0,1].
        mf_values = np.clip(mf_values, 0.0, 1.0)

        return mf_values

    def calc_cond_probs(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates conditional probabilities P(class|cluster) with multi-class support.
        These probabilities are aligned with the actual risk of the GMM clusters.
        """
        mf_vals = self.calc_mf(X, normalized=True) # Uses GMM membership probabilities
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        if mf_vals.shape[0] == 0 or y.size == 0 or n_classes < 2 or self.n_clusters == 0:
            logger.warning("PFLZ.calc_cond_probs: Insufficient or single-class data. Returning uniform probabilities.")
            # Ensure correct dimension for the return even in case of error or empty data
            n_clusters_fallback = self.n_clusters if self.n_clusters > 0 else 1
            n_classes_fallback = max(2, n_classes) # Minimum 2 classes for binary probabilities
            return np.full((n_clusters_fallback, n_classes_fallback), 1.0 / n_classes_fallback)

        cond_probs = np.zeros((self.n_clusters, n_classes)) # mf_vals.shape[1] is n_clusters

        for cls_idx, cls_val in enumerate(unique_classes):
            class_mask = (y == cls_val)
            numerator = np.sum(mf_vals[class_mask], axis=0) # Sum of MFs for samples of this class

            denominator = np.sum(mf_vals, axis=0) # Total sum of MFs per cluster
            denominator = np.where(denominator < self.epsilon, self.epsilon, denominator) # Division by zero protection

            cond_probs[:, cls_idx] = numerator / denominator

        # Handle missing/infinite values
        cond_probs = np.nan_to_num(cond_probs, nan=0.5, posinf=1.0, neginf=0.0)

        # Final normalization so that the sum per row (per cluster) equals 1
        row_sum = np.sum(cond_probs, axis=1, keepdims=True)
        cond_probs = cond_probs / np.where(row_sum > self.epsilon, row_sum, 1)

        return cond_probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PFLZ":
        """
        Trains the PFL model with error handling.

        Steps:
        1. Train the GMM for probabilistic clustering.
            - Uses KMeans to initialize GMM means (for greater stability).
            - The GMM covariance type is set to 'diag' to adjust membership functions.
        2. Calculate conditional probabilities (P(class|cluster)).
        3. Create fuzzy features (GMM memberships) and interpretability features.
        4. Train LightGBM on original and fuzzy features.
        5. Train a fusion model (LogisticRegression) to combine the LightGBM prediction
            and the dominant fuzzy rule probability.
        """
        self.X_original_fit = X # Stores X at the start of fit

        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            logger.warning("PFLZ.fit: Insufficient X or y data to train the model. Skipping training.")
            # Initialize necessary attributes so that predict_proba doesn't raise an error
            self.lgb_model = None
            self.fusion_model = None
            self.cond_probs_ = np.full((self.n_clusters, 2), 0.5) # Default conditional probs
            self.cluster_centers_ = np.array([]) # Ensure it's empty or a placeholder
            self.n_original_features_ = X.shape[1] if X.shape[0] > 0 else 0
            self.n_fuzzy_features_ = self.n_clusters
            return self

        try:
            # 1. Train the GMM for probabilistic clustering
            # Ensure there are enough samples for GMM
            if X.shape[0] < self.n_clusters:
                logger.warning(f"Insufficient number of samples ({X.shape[0]}) for GMM with {self.n_clusters} clusters. Reducing the number of GMM components.")
                n_comp_gmm = max(1, X.shape[0]) # At least 1 component
            else:
                n_comp_gmm = self.n_clusters

            # Use KMeans to initialize GMM means for greater stability
            # Ensure KMeans also has enough samples
            if X.shape[0] >= n_comp_gmm:
                initial_means = KMeans(n_clusters=n_comp_gmm, random_state=42, n_init='auto', max_iter=300).fit(X).cluster_centers_
            else:
                logger.warning("Not enough samples to initialize KMeans for GMM. Random initialization of GMM means.")
                initial_means = np.random.rand(n_comp_gmm, X.shape[1]) # Fallback if X.shape[0] is very small

            # Adjusting fuzzy membership functions: Use covariance_type='diag'
            self.cluster_model = GaussianMixture(n_components=n_comp_gmm, random_state=42, means_init=initial_means, covariance_type='diag') # ADDED: covariance_type='diag'
            self.cluster_model.fit(X)
            # Save the centroids (means of GMM components)
            self.cluster_centers_ = self.cluster_model.means_

            # 2. Calculate conditional probabilities (P(class|cluster))
            # This is now based on GMM memberships
            self.cond_probs_ = self.calc_cond_probs(X, y)

            # 3. Create fuzzy features (GMM memberships)
            fuzzy_features = self.calc_mf(X, normalized=True)

            # Handle the case where fuzzy_features is empty after calc_mf
            if fuzzy_features.size == 0:
                logger.warning("PFLZ.fit: Fuzzy features are empty after calc_mf. LightGBM and fusion will not be trained.")
                self.lgb_model = None
                self.fusion_model = None
                return self

            # Prepare the feature for the dominant fuzzy rule (mortality probability of the dominant cluster)
            dominant_rule_prob = np.zeros(X.shape[0])
            if self.cond_probs_ is not None and self.cond_probs_.shape[1] > 1:
                # Get the index of the dominant cluster for each sample
                dominant_cluster_indices = np.argmax(fuzzy_features, axis=1)
                # Retrieve the mortality probability for the dominant cluster of each sample
                dominant_rule_prob = self.cond_probs_[dominant_cluster_indices, 1]
            else:
                logger.warning("Conditional probabilities (P(class|cluster)) not available or single-class. The dominant rule feature will be 0.5.")
                dominant_rule_prob = np.full(X.shape[0], 0.5) # Fallback

            # 4. Train LightGBM on original and fuzzy features (GMM memberships)
            X_extended_lgbm = np.concatenate((X, fuzzy_features), axis=1)

            lgb_params_final = self.lgb_params.copy()
            # Ensure binary objective and number of classes if not defined
            # ATTENTION: If the objective is 'binary', LightGBM DOES NOT ACCEPT 'num_classes'.
            # We remove it if the objective is binary.
            if lgb_params_final.get('objective') == 'binary':
                lgb_params_final.pop('num_classes', None)
            elif 'num_classes' not in lgb_params_final: # For multi-class objectives if needed
                lgb_params_final['num_classes'] = len(np.unique(y)) if len(np.unique(y)) > 1 else 1


            # Apply monotonicity constraints if "feature_names" is provided and matches
            if self.feature_names is not None:
                if 'monotone_constraints' in lgb_params_final:
                    if len(lgb_params_final['monotone_constraints']) == X.shape[1]:
                        lgb_params_final['monotone_constraints'].extend([0] * fuzzy_features.shape[1])
                    else:
                        logger.warning("Monotonicity constraints provided in lgb_params do not match the number of original features. They will be ignored.")
                        lgb_params_final.pop('monotone_constraints', None) # Remove invalid constraints
                else:
                    lgb_params_final.pop('monotone_constraints', None) # Remove monotonicity constraints

            self.lgb_model = lgb.LGBMClassifier(**lgb_params_final)
            self.lgb_model.fit(X_extended_lgbm, y)

            # 5. Train a fusion model (LogisticRegression) to combine probabilities
            lgbm_proba = self.lgb_model.predict_proba(X_extended_lgbm)[:, 1] # Class 1 probability

            # Create the dataset for the fusion model
            # Input: [lgbm_proba, dominant_rule_proba]
            X_fusion = np.column_stack((lgbm_proba, dominant_rule_prob))

            # Fusion model (for example, logistic regression, simple and interpretable)
            self.fusion_model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
            self.fusion_model.fit(X_fusion, y)

            self.n_original_features_ = X.shape[1]
            self.n_fuzzy_features_ = fuzzy_features.shape[1] # Number of clusters

            return self

        except Exception as e:
            logger.error(f"Error during PFL training: {str(e)}")
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities using LightGBM and the fusion model.
        """
        if self.lgb_model is None or self.fusion_model is None or self.cluster_model is None:
            logger.error("PFLZ model has not been trained or is incomplete. Returning uniform probabilities.")
            return np.full((X.shape[0], 2), 0.5) if X.shape[0] > 0 else np.array([])

        if X.shape[0] == 0:
            return np.array([]) # Return an empty array if X is empty

        # Get GMM membership probabilities
        fuzzy_features = self.calc_mf(X, normalized=True)
        if fuzzy_features.size == 0: # If calc_mf returns empty
            logger.warning("Fuzzy features are empty during prediction. Returning 0.5 for all probabilities.")
            return np.full((X.shape[0], 2), 0.5)

        # Extend X for LightGBM
        # Ensure that the number of features of X matches what LightGBM expects
        # This is a robustness check, as dimensions should be consistent after fit
        if X.shape[1] != self.n_original_features_ or fuzzy_features.shape[1] != self.n_fuzzy_features_:
             logger.error(f"Feature count inconsistency: X has {X.shape[1]} original features, PFLZ expects {self.n_original_features_}. Or fuzzy features have {fuzzy_features.shape[1]} while PFLZ expects {self.n_fuzzy_features_}. Returning uniform probabilities.")
             return np.full((X.shape[0], 2), 0.5)

        X_extended_lgbm = np.concatenate((X, fuzzy_features), axis=1)

        # LightGBM prediction
        # Handle the case where lgbm_model.predict_proba returns a 1D array (for a single class)
        try:
            lgbm_probas_all_classes = self.lgb_model.predict_proba(X_extended_lgbm)
            # Extract the probability of the positive class (assumed to be 1)
            # lgbm_probas_all_classes[:, 1] is safe if the objective is 'binary'
            lgbm_proba_positive_class = lgbm_probas_all_classes[:, 1]
        except Exception as e:
            logger.error(f"Error during LightGBM prediction: {e}. Returning 0.5 for LightGBM proba.")
            lgbm_proba_positive_class = np.full(X.shape[0], 0.5)


        # Calculate the probability of the dominant fuzzy rule
        dominant_rule_prob = np.zeros(X.shape[0])
        if self.cond_probs_ is not None and self.cond_probs_.shape[1] > 1:
            dominant_cluster_indices = np.argmax(fuzzy_features, axis=1)
            dominant_rule_prob = self.cond_probs_[dominant_cluster_indices, 1]
        else:
            logger.warning("Conditional probabilities (P(class|cluster)) not available or single-class. The dominant rule feature will be 0.5.")
            dominant_rule_prob = np.full(X.shape[0], 0.5)

        # Prepare input for the fusion model
        X_fusion = np.column_stack((lgbm_proba_positive_class, dominant_rule_prob))

        # Prediction of the fusion model (class 1 probabilities)
        try:
            fused_proba_all_classes = self.fusion_model.predict_proba(X_fusion)
            # Return probabilities [proba_class_0, proba_class_1]
            return fused_proba_all_classes
        except Exception as e:
            logger.error(f"Error during fusion model prediction: {e}. Returning uniform probabilities.")
            return np.full((X.shape[0], 2), 0.5)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts classes using the fusion model.
        """
        if X.shape[0] == 0:
            return np.array([])
        # predict_proba returns (n_samples, n_classes), so argmax on the class axis
        return np.argmax(self.predict_proba(X), axis=1)