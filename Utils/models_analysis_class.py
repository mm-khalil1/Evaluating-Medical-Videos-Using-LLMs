import os
import itertools

from .statistics_plots_analysis_utils import *

class ModelsAnalysis:
    """
    A class to analyze models' performance based on expert ratings and model ratings.

    Usage:
        # Example usage of ModelAnalysis class
        ```
        model_files_dict = {'GPT-4': 'gpt4_responses.csv', 'Gemini': 'gemini_responses.csv}
        experts_file = 'path/to/experts_data.csv'
        models_dir = 'path/to/models'
        topics_keys = ['SB', 'FF', 'CH', 'TF', 'PN', 'ISA']
        categories = [1, 2, 3, 4, 5]
        agreement_coef = 'Brennan-Prediger Kappa'
        weights_type = 'quadratic'

        analysis = ModelAnalysis(model_files_dict, experts_file, models_dir, topics_keys,
                                 categories, agreement_coef, weights_type)

        analysis.load_experts_data()
        analysis.process_models()

        total_experts_agreement = analysis.calculate_total_experts_agreement()
        qw_experts_agreement = analysis.calculate_qw_experts_agreement()

        descriptive_stats = analysis.generate_descriptive_stat()
        total_models_agreement = analysis.calculate_total_expert_models_agreement()
        qw_models_agreement = analysis.calculate_qw_expert_models_agreement()
        ```
    """

    def __init__(self, model_files_dict, experts_file, models_dir, topics_keys,
                 categories, agreement_coef, weights_type, question_columns=None, return_cols=None):
        """
        Initializes the ModelAnalysis instance with provided parameters.

        Args:
            model_files_dict (dict): A dictionary mapping model names to their respective CSV file paths.
            experts_file (str): Path to the CSV file containing expert ratings data.
            models_dir (str): Directory path where model CSV files are stored.
            topics_keys (list): List of topic keys for filtering expert data.
            categories (list): List of ordinal categories used in rating.
            agreement_coef (str): Coefficient name used for agreement calculation.
            weights_type (str): Type of weights used in agreement calculations.
            question_columns (list, optional): List of question column names, e.g., ['Q1', 'Q2', ..., 'Q15'].
            return_cols (list, optional): Columns to return from experts_df if specified; default is all columns.
        """
        self.model_files_dict = model_files_dict
        self.model_names = list(model_files_dict.keys())
        self.experts_df = pd.read_csv(experts_file)
        self.models_dir = models_dir
        self.topics_keys = topics_keys
        if return_cols is None:
            self.return_cols = self.experts_df.columns
        else:
            self.return_cols = return_cols
        self.agreement_coef = agreement_coef
        self.weights_type = weights_type
        if question_columns is None:
            self.question_columns = QUESTIONS_COLUMNS
        else:
            self.question_columns = question_columns
        self.categories = categories
        num_questions = len(self.question_columns)
        self.total_score_categories = list(range(min(categories) * num_questions,
                                                 max(categories) * num_questions + 1))

        self.group_df = None
        self.total_expert_models_stat = None

    def load_experts_data(self):
        """
        Preprocesses expert ratings data.

        Splits expert DataFrame on topic groups based on topics_keys and calculates average ratings.
        """
        self.group_df = filter_df_by_topics(self.experts_df, TOPICS, self.topics_keys, return_cols=self.return_cols)
        self.group_df = calculate_experts_avg_of_questions(self.group_df, EXPERT1_COLUMNS, EXPERT2_COLUMNS)

    def _process_model_group(self, model_name, model_df, questions_to_sum, binarize=False) -> pd.DataFrame:
        """
        Processes model data and integrates it with the expert group DataFrame.

        Args:
            model_name (str): Name of the model.
            model_df (pd.DataFrame): DataFrame containing model data.
            questions_to_sum (list): List of questions to sum for processing.
            binarize (bool): Whether to binarize scores.

        Returns:
            pd.DataFrame: Updated DataFrame with processed model data.
        """
        # Filter model_df to include only rows where 'Video ID' is in group_df['Video ID']
        model_group_df = model_df[model_df['Video ID'].isin(self.group_df['Video ID'])].reset_index(drop=True)

        # Binarize scores if specified. This is for videos where expert scores in the range [0, 1]
        if binarize:
            for column in questions_to_sum:
                model_group_df[column] = model_group_df[column].apply(binarize_value)

        # Calculate and insert total score for the specified columns
        model_group_df[model_name] = model_group_df[questions_to_sum].sum(axis=1)

        # Rename model score columns to include the model name as a prefix
        new_column_names = {question: f'{model_name} {question}' for question in questions_to_sum}
        model_group_df.rename(columns=new_column_names, inplace=True)

        # Merge group_df with model_group_df on the specified columns
        columns_to_merge = ['Video ID', model_name] + list(new_column_names.values())
        self.group_df = merge_dataframes(self.group_df, model_group_df, columns_to_merge)

        return self.group_df
    
    def process_models(self):
        """
        Processes model ratings from CSV files and integrates them with expert ratings.
        """
        for model_name, model_file in self.model_files_dict.items():
            model_path = os.path.join(self.models_dir, model_file)
            model_df = pd.read_csv(model_path, encoding='utf-8', usecols=['Video ID'] + self.question_columns)
            self.group_df = self._process_model_group(model_name, model_df, questions_to_sum=self.question_columns)

    def _calculate_total_statistics(self, agreement_coef=None, weights_type='quadratic') -> pd.DataFrame:
        """
        Private method to calculate statistical tests for experts' average scores and model scores.

        Args:
            agreement_coef (str, optional): Specific agreement coefficient. Default is None.
            weights_type (str, optional): Type of weights to use. Default is 'quadratic'.

        Returns:
            pd.DataFrame: DataFrame containing the calculated statistical tests.
        """
        rater1_columns = ['Experts_Avg'] * len(self.model_names)
        return calculate_stat_df(self.group_df, rater1_columns, self.model_names,
                                 self.total_score_categories, weights_type=weights_type)

    def calculate_total_experts_agreement(self, group_df=None) -> float:
        """
        Calculates total agreement between Expert1 and Expert2 on specified categories.

        Args:
            group_df (pd.DataFrame, optional): DataFrame containing grouped expert ratings data. If None,
                                               uses self.group_df.

        Returns:
            float: Total agreement score based on agreement_coef.
        """
        if group_df is None:
            group_df = self.group_df
        experts_agreement = calculate_statistics(group_df, 'Expert1', 'Expert2',
                                                 self.total_score_categories, self.weights_type)
        return experts_agreement.get(self.agreement_coef)

    def calculate_qw_experts_agreement(self, group_df=None) -> pd.DataFrame:
        """
        Calculates question-wise agreement between Expert1 and Expert2 for each question.

        Args:
            group_df (pd.DataFrame, optional): DataFrame containing grouped expert ratings data. If None,
                                               uses self.group_df.

        Returns:
            pd.DataFrame: DataFrame with question-wise agreement scores for each question.
        """
        if group_df is None:
            group_df = self.group_df
        return calculate_stat_df(group_df, EXPERT1_COLUMNS, EXPERT2_COLUMNS,
                                 self.categories, self.agreement_coef, self.weights_type,
                                 output_df_column_names=self.question_columns)

    def generate_descriptive_stat(self) -> pd.DataFrame:
        """
        Generates descriptive statistics for experts' average scores and model scores.

        Returns:
            pd.DataFrame: DataFrame with mean and standard deviation of Experts_Avg and each model's scores.
        """
        descriptive_stat_df = self.group_df.copy().describe().round(2).T
        descriptive_stat_df.rename_axis('Model', axis=1, inplace=True)

        rows_to_keep = ['Experts_Avg'] + self.model_names
        descriptive_stat_df = descriptive_stat_df.loc[rows_to_keep, ['mean', 'std']]
        
        # stat_to_add = ['Paired t-test', "Cohen's d"]#, 'Wilcoxon signed-rank test p-value', 'Effect Size']
        # statistical_tests = self.total_expert_models_stat.T[stat_to_add]
        # descriptive_stat_df = pd.concat([descriptive_stat_df, statistical_tests], axis=1)

        return descriptive_stat_df

    def remove_distant_ratings(self, model_score_columns, expert1_columns, expert2_columns, max_diff=2) -> pd.DataFrame:
        '''
        Remove distant ratings from the DataFrame based on the difference between expert ratings.

        Parameters:
            df (pd.DataFrame): DataFrame containing the ratings.
            model_score_columns (list): List of model score column names.
            expert1_columns (list): List of Expert 1 column names.
            expert2_columns (list): List of Expert 2 column names.
            max_diff (int): Maximum allowed difference between Expert 1 and Expert 2 ratings.

        Returns:
            pd.DataFrame: DataFrame with distant ratings removed.
        '''
        cleaned_experts_df = self.group_df.copy()
        num_of_distant_ratings_per_q = {}

        for q_num, col1, col2 in zip(model_score_columns, expert1_columns, expert2_columns):
            # Count videos with difference between expert1 and expert2 greater than max_diff
            df_big_diff = cleaned_experts_df[abs(cleaned_experts_df[col1] - cleaned_experts_df[col2]) > max_diff]
            num_of_distant_ratings = df_big_diff['Video ID'].tolist()
            num_of_distant_ratings_per_q[q_num] = len(num_of_distant_ratings)
            # print(f'{q_num}: Length of videos with distant ratings:', len(num_of_distant_ratings))
            
            cleaned_experts_df.loc[cleaned_experts_df['Video ID'].isin(num_of_distant_ratings), col1] = np.nan
            cleaned_experts_df.loc[cleaned_experts_df['Video ID'].isin(num_of_distant_ratings), col2] = np.nan

        # Assuming calculate_experts_avg_of_questions is defined elsewhere and takes the necessary parameters
        cleaned_experts_df = calculate_experts_avg_of_questions(cleaned_experts_df, expert1_columns, expert2_columns)
        
        return cleaned_experts_df, num_of_distant_ratings_per_q

    def calculate_total_expert_models_agreement(self, models_in_order=False, ascending=False) -> pd.DataFrame:
        """
        Calculates total score agreement between experts and each model.

        Args:
            models_in_order (bool, optional): If True, order the models according to the agreement 
                                              measure in descending order. Default is False.

        Returns:
            pd.DataFrame: A DataFrame with total agreement scores sorted by the specified agreement 
                          measure.
        """
        self.total_expert_models_stat = self._calculate_total_statistics(self, weights_type=self.weights_type)

        if models_in_order:
            self.total_expert_models_stat = self.total_expert_models_stat.T.sort_values(by=self.agreement_coef, ascending=ascending).T
            self.model_names = self.total_expert_models_stat.columns.tolist()
            
        return self.total_expert_models_stat.loc[[self.agreement_coef]]

    def calculate_qw_expert_models_agreement(self, group_df=None) -> pd.DataFrame:
        """
        Calculates question-wise score agreement between experts and each model for each question.

        Args:
            group_df (pd.DataFrame, optional): DataFrame containing grouped expert ratings data. If None,
                                               uses self.group_df.

        Returns:
            pd.DataFrame: DataFrame with question-wise agreement scores for each model and each question.
        """
        if group_df is None:
            group_df = self.group_df

        qw_agreement_df = pd.DataFrame(index=self.question_columns, columns=self.model_names, dtype=float)
        for model_name in self.model_names:
            for question_num, expert_col in enumerate(EXPERTS_AVG_COLUMNS, start=1):
                model_col = f'{model_name} Q{question_num}'
                stat = calculate_statistics(group_df, expert_col, model_col,
                                            self.categories, self.weights_type)
                qw_agreement_df.at[f'Q{question_num}', model_name] = round(stat.get(self.agreement_coef), 2)

        return qw_agreement_df
    
    def calculate_models_agreement(self) -> pd.DataFrame:
        """
        Calculates the agreement between models and experts.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame with agreement scores.
        """
        models_agreement_df = pd.DataFrame(index=self.model_names[:-1], columns=self.model_names[1:], dtype=float)

        # Fill only the upper triangular part of the DataFrame with the 'agreement_coef' values
        for model1, model2 in itertools.combinations(self.model_names, 2):
            stat = calculate_statistics(self.group_df, model1, model2, self.total_score_categories, self.weights_type)
            models_agreement_df.at[model1, model2] = stat.get(self.agreement_coef)
        
        return models_agreement_df
