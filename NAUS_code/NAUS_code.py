#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
#Function for data undersampling with algorithm NAUS
#Output data: undersampled dataset
class DataProcessor:
    def __init__(self, data, class_col, feature_cols):
        """
        Initialize the DataProcessor class.

        Parameters:
            data (pd.DataFrame): The dataset.
            class_col (str): The column name for class labels.
            feature_cols (list of str): List of column names for features.
        """
        self.data = data
        self.class_col = class_col
        self.maj_class=None
        self.feature_cols = feature_cols
        self.filtered_data = None
        self.noise = None
        self.undersampled_data = None
        self.delated = None
#        self.data_plot=None

    def calculate_posterior_probabilities(self):
        """
        Calculate posterior probabilities for the dataset using Bayes' theorem.
        """
        classes = self.data[self.class_col].unique()
        if len(classes) != 2:
            raise ValueError("This method supports binary classification only.")
        
        class_0, class_1 = classes

        # Compute priors
        prior_0 = len(self.data[self.data[self.class_col] == class_0]) / len(self.data)
        prior_1 = len(self.data[self.data[self.class_col] == class_1]) / len(self.data)

        # Compute means and variances for each class and feature
        stats_0 = self.data[self.data[self.class_col] == class_0][self.feature_cols].agg(['mean', 'var'])
        stats_1 = self.data[self.data[self.class_col] == class_1][self.feature_cols].agg(['mean', 'var'])

        # Compute posterior probabilities
        def posterior_probability(row):
            likelihood_0 = np.prod([
                np.exp(-0.5 * ((row[col] - stats_0.at['mean', col]) ** 2) / stats_0.at['var', col]) /
                np.sqrt(2 * np.pi * stats_0.at['var', col])
                for col in self.feature_cols
            ])
            likelihood_1 = np.prod([
                np.exp(-0.5 * ((row[col] - stats_1.at['mean', col]) ** 2) / stats_1.at['var', col]) /
                np.sqrt(2 * np.pi * stats_1.at['var', col])
                for col in self.feature_cols
            ])

            posterior_0 = prior_0 * likelihood_0
            posterior_1 = prior_1 * likelihood_1

            if posterior_0 + posterior_1 == 0:  # Avoid division by zero
                return 0.5
            return posterior_1 / (posterior_0 + posterior_1)

        self.data['p_other_class'] = self.data.apply(posterior_probability, axis=1)

    def compute_tacf(self, threshold_type, threshold_value, min_features=2):
        """
        Apply the Threshold-Adjusted Classification Filter (TACF) algorithm.

        Parameters:
            threshold_type (str): Type of threshold to use, either 'percent' or 'std'.
            threshold_value (float): Value of the threshold (e.g., 10 for percent or 1.5 for std).
            min_features (int): Minimum number of features exceeding the threshold to classify as noise.
        """
        classes = self.data[self.class_col].unique()
        if len(classes) != 2:
            raise ValueError("TACF currently supports binary classification only.")

        class_p, class_n = classes
        data_p = self.data[self.data[self.class_col] == class_p].copy()
        data_n = self.data[self.data[self.class_col] == class_n].copy()

        if threshold_type == 'percent':
            p_threshold_p = np.percentile(data_p['p_other_class'], 100 - threshold_value)
            p_threshold_n = np.percentile(data_n['p_other_class'], 100 - threshold_value)

            noisy_p = data_p[data_p['p_other_class'] > p_threshold_p]
            noisy_n = data_n[data_n['p_other_class'] > p_threshold_n]

            data_p = data_p[data_p['p_other_class'] <= p_threshold_p]
            data_n = data_n[data_n['p_other_class'] <= p_threshold_n]

        elif threshold_type == 'std':
            stats_p = data_p[self.feature_cols].agg(['mean', 'std'])
            stats_n = data_n[self.feature_cols].agg(['mean', 'std'])

            def is_noisy(row, stats, threshold_value, min_features):
                count = sum(
                    abs(row[col] - stats.at['mean', col]) > threshold_value * stats.at['std', col]
                    for col in self.feature_cols
                )
                return count >= min_features

            noisy_p = data_p[data_p.apply(lambda row: is_noisy(row, stats_p, threshold_value, min_features), axis=1)]
            noisy_n = data_n[data_n.apply(lambda row: is_noisy(row, stats_n, threshold_value, min_features), axis=1)]

            data_p = data_p[~data_p.index.isin(noisy_p.index)]
            data_n = data_n[~data_n.index.isin(noisy_n.index)]

        else:
            raise ValueError("Invalid threshold_type. Use 'percent' or 'std'.")

        self.filtered_data = pd.concat([data_p, data_n]).sort_index()
        self.noise = pd.concat([noisy_p, noisy_n]).sort_index()
    def _calculate_potential(self, x, majority_class, minority_class, gamma):
        """
        Calculate the mutual class potential for a point x.

        Parameters:
            x (array-like): The point for which to calculate the potential.
            majority_class (array-like): Majority class observations.
            minority_class (array-like): Minority class observations.
            gamma (float): Spread parameter for the RBF.

        Returns:
            float: Mutual class potential at point x.
        """
        dist_majority = euclidean_distances([x], majority_class)
        dist_minority = euclidean_distances([x], minority_class)

        potential_majority = np.sum(np.exp(-(dist_majority / gamma) ** 2))
        potential_minority = np.sum(np.exp(-(dist_minority / gamma) ** 2))

        return potential_majority - potential_minority

    def mutual_class_potential_undersample(self, gamma=1.0):
        """
        Undersample the majority class using mutual class potential.
        
        Parameters:
            gamma (float): Spread parameter for the RBF.
        
        Returns:
            pd.DataFrame: Undersampled dataset with original indices.
            pd.DataFrame: Deleted dataset with original indices.
        """
        if self.undersampled_data is None:
            raise ValueError("Noise removal must be performed before undersampling.")
        classes = self.data[self.class_col].unique()
        # Extract majority and minority class samples with their original indices
        majority_class = self.undersampled_data[self.undersampled_data[self.class_col] == self.maj_class]
        minority_class = self.undersampled_data[self.undersampled_data[self.class_col] == classes[1] if classes[0] == self.maj_class else classes[0]]
        
        # Store the original indices of the majority class samples
        majority_indices = majority_class.index
        majority_class_values = majority_class[self.feature_cols].values
        minority_class_values = minority_class[self.feature_cols].values
        
        # Calculate mutual class potential for each majority sample
        potentials = [
            self._calculate_potential(x, majority_class_values, minority_class_values, gamma)
            for x in majority_class_values
        ]
        
        # Determine the number of samples to remove
        n_majority = len(majority_class_values)
        n_minority = len(minority_class_values)
        n_to_remove = n_majority - 2 * n_minority  # Remove until majority is <= 2 * minority
        if n_to_remove > 0:
            # Get the indices of the majority class samples to remove based on potential (highest first)
            remove_indices = []
            for _ in range(n_to_remove):
                # Find the index of the majority sample with the highest potential
                highest_potential_index = np.argmax(potentials)
                remove_indices.append(majority_indices[highest_potential_index])
                potentials[highest_potential_index] = -np.inf  # Mark this sample as removed by setting its potential to -inf
        
            # Get the indices of the majority class samples to keep
            keep_indices = majority_indices.difference(remove_indices)
        
            # Create the undersampled majority class
            undersampled_majority = self.undersampled_data.loc[keep_indices]
            delated_majority = self.undersampled_data.loc[remove_indices]
        else:
            # No samples to remove
            undersampled_majority = majority_class
            delated_majority = pd.DataFrame(columns=self.undersampled_data.columns)
        
        # Combine the undersampled majority class with the minority class
        undersampled_data = pd.concat([undersampled_majority, minority_class])
        delated_data = delated_majority
        
        return undersampled_data, delated_data

    def local_gran_subspaces(self, dataset):
        """
        Generate local granular subspaces by dropping one feature at a time.

        Parameters:
            dataset (pd.DataFrame): The dataset to process.

        Returns:
            list: List of subspaces.
        """
        subsp_list = []
        for i in dataset.columns:
            subsp = dataset.drop(columns=[i])
            subsp_list.append(subsp)
        return subsp_list

    def calculate_and_sort_distances(self, dataset, distance_type="euclidean"):
        """
        Calculate distances (Mahalanobis or Euclidean) and sort the dataset.

        Parameters:
            dataset (pd.DataFrame): The dataset to process.
            distance_type (str): Type of distance to use, either 'euclidean' or 'mahalanobis'.

        Returns:
            pd.DataFrame: Sorted dataset with distances.
        """
        X = dataset.iloc[:, :-1]
        mean_vector = X.mean(axis=0)
        distances = []
        if distance_type == "mahalanobis":
            cov_matrix = np.cov(X, rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            for _, row in X.iterrows():
                distances.append(mahalanobis(row, mean_vector, inv_cov_matrix))
        elif distance_type == "euclidean":
            for _, row in X.iterrows():
                distances.append(euclidean(row, mean_vector))
        else:
            raise ValueError("Unsupported distance type. Use 'euclidean' or 'mahalanobis'.")
        dataset_with_distances = dataset.copy()
        dataset_with_distances['distance'] = distances
        sorted_dataset = dataset_with_distances.sort_values(by='distance').reset_index(drop=True)
        return sorted_dataset

    def mark_tomek_links(self, dataset, labels, maj):
        """
        Mark Tomek links in the dataset.

        Parameters:
            dataset (pd.DataFrame): The dataset to process.
            labels (pd.Series): The class labels.
            maj: The majority class label.

        Returns:
            pd.DataFrame: Dataset with Tomek links marked.
        """
        nbrs = NearestNeighbors(n_neighbors=2).fit(dataset)
        distances, indices = nbrs.kneighbors(dataset)
        tomek_link_mark = np.zeros(len(dataset), dtype=int)
        for i in range(len(dataset)):
            neighbor_index = indices[i, 1]
            if labels.iloc[i] != labels.iloc[neighbor_index]:
                if labels.iloc[i] == maj:
                    tomek_link_mark[i] = 1
                elif labels.iloc[neighbor_index] == maj:
                    tomek_link_mark[neighbor_index] = 1
        result_dataset = dataset.copy()
        result_dataset['tomek_link'] = tomek_link_mark
        return result_dataset, tomek_link_mark

    def combine_tomek_marks(self, dataset, labels, maj):
        """
        Combine Tomek marks from all subspaces.

        Parameters:
            dataset (pd.DataFrame): The dataset to process.
            labels (pd.Series): The class labels.
            maj: The majority class label.

        Returns:
            pd.DataFrame: Combined Tomek marks.
        """
        subspaces = self.local_gran_subspaces(dataset)
        combined_tomek_marks = pd.DataFrame(index=dataset.index)
        for i in range(len(subspaces)):
            _, tomek_mark = self.mark_tomek_links(subspaces[i], labels, maj)
            combined_tomek_marks[f'tomek_link_{dataset.columns[i]}'] = tomek_mark
        return combined_tomek_marks

    def undersample(self, maj_class, gamma=1.0, ratio = 0.5):
        """
        Undersample the majority class using Tomek links and mutual class potential.

        Parameters:
            maj_class: The majority class label.
            gamma (float): Spread parameter for the RBF.
        """
        if self.filtered_data is None:
            raise ValueError("Noise removal must be performed before undersampling.")
        self.maj_class=maj_class
        # Step 1: Undersample using Tomek links
        combined_marks = self.combine_tomek_marks(self.filtered_data[self.feature_cols], self.filtered_data[self.class_col], maj_class)
        combined_marks['total'] = combined_marks.sum(axis=1)

        to_remove = combined_marks[combined_marks['total'] > 0].index
        not_removed = combined_marks[combined_marks['total'] <= 0].index
        self.undersampled_data = self.filtered_data.drop(to_remove)
        self.delated = self.filtered_data.drop(not_removed)

        # Step 2: Check if majority class is still more than twice the minority class
        n_majority = len(self.undersampled_data[self.undersampled_data[self.class_col] == maj_class])
        n_minority = len(self.undersampled_data[self.undersampled_data[self.class_col] != maj_class])
        
        if ratio*n_majority > n_minority:
            # Apply mutual class potential undersampling
            self.undersampled_data, delated = self.mutual_class_potential_undersample(gamma)
            delated_df = delated # Выбираем строки по индексам
            delated_df[self.class_col] = maj_class  # Ensure class column is included
            if self.delated is None:
                self.delated = delated_df
            else:
                self.delated = pd.concat([self.delated, delated_df])
        return self.undersampled_data

    def visualize(self, title="UMAP Projection",mode='noise'):
        """
        Visualize the dataset using UMAP.

        Parameters:
            title (str): Title of the plot.
        """
        if self.filtered_data is None:
            raise ValueError("Noise removal must be performed before visualization.")
        # Use filtered_data or undersampled_data if available
        if mode == 'noise':
            data = self.data
        else:
            data = self.filtered_data
        reducer = umap.UMAP(random_state=0,n_neighbors=5)
        umap_embeddings = reducer.fit_transform(data[self.feature_cols])
        # Add UMAP embeddings to the dataset
        data["UMAP1"] = umap_embeddings[:, 0]
        data["UMAP2"] = umap_embeddings[:, 1]
        majority_class = data[self.class_col].mode()[0]
        if mode == 'noise':
            filtered_data=self.filtered_data
            noise = self.noise
            filtered_data["UMAP1"] = data.loc[self.filtered_data.index, "UMAP1"]
            filtered_data["UMAP2"] = data.loc[self.filtered_data.index, "UMAP2"]
            noise["UMAP1"] = data.loc[self.noise.index, "UMAP1"]
            noise["UMAP2"] = data.loc[self.noise.index, "UMAP2"]
        else:
            filtered_data=self.undersampled_data
            noise = self.delated
            filtered_data["UMAP1"] = data.loc[self.undersampled_data.index, "UMAP1"]
            filtered_data["UMAP2"] = data.loc[self.undersampled_data.index, "UMAP2"]
            noise["UMAP1"] = data.loc[self.delated.index, "UMAP1"]
            noise["UMAP2"] = data.loc[self.delated.index, "UMAP2"]
        
        # Plot scatterplot
        plt.figure(figsize=(10, 8), dpi=120)
        # Plot majority and minority points
        if mode == 'noise':
            # Plot noisy points as crosses
            plt.scatter(
                filtered_data.loc[filtered_data[self.class_col] == majority_class, "UMAP1"],
                filtered_data.loc[filtered_data[self.class_col] == majority_class, "UMAP2"],
                c="lightblue", s = 100, alpha=0.6, label="Majority Class", edgecolor="k", linewidth=0.5
            )
            
            plt.scatter(
                filtered_data.loc[filtered_data[self.class_col] != majority_class, "UMAP1"],
                filtered_data.loc[filtered_data[self.class_col] != majority_class, "UMAP2"],
                c="yellow",s =100, alpha=0.6, label="Minority Class", edgecolor="k", linewidth=0.5
            )
            plt.scatter(
                noise.loc[noise[self.class_col] == majority_class, "UMAP1"],
                noise.loc[noise[self.class_col] == majority_class, "UMAP2"],
                c="blue",s = 100, alpha=0.8, label="Noisy Majority", marker="x"
            )
            
            plt.scatter(
                noise.loc[noise[self.class_col] != majority_class, "UMAP1"],
                noise.loc[noise[self.class_col] != majority_class, "UMAP2"],
                c="orange",s = 100, alpha=0.8, label="Noisy Minority", marker="x"
            )
        elif mode == 'undersampling':
            plt.scatter(
                noise["UMAP1"],
                noise["UMAP2"],
                c="black",s = 20, alpha=0.6, label="Deleted data", marker="x"
            ) 
            plt.scatter(
                filtered_data.loc[filtered_data[self.class_col] == majority_class, "UMAP1"],
                filtered_data.loc[filtered_data[self.class_col] == majority_class, "UMAP2"],
                c="lightblue",s = 100, alpha=0.6, label="Majority Class", edgecolor="k", linewidth=0.5
            )
            
            plt.scatter(
                filtered_data.loc[filtered_data[self.class_col] != majority_class, "UMAP1"],
                filtered_data.loc[filtered_data[self.class_col] != majority_class, "UMAP2"],
                c="yellow",s = 100, alpha=0.6, label="Minority Class", edgecolor="k", linewidth=0.5
            )

        plt.legend(fontsize=12, loc="best")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.title("")
        plt.grid(False)
        plt.show()

