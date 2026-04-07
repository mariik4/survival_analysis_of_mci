import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sksurv.util import Surv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_MISSINGNESS_THRESHOLD   = 20
HIGH_MISSINGNESS_THRESHOLD  = 80

SURVIVAL_EVENT_COL   = "EVENT_MCI"
SURVIVAL_TIME_COL    = "TIME"
MANDATORY_FEATURE    = "categorical__group_missing_indicator_0"

NOT_COLLECTED_PLACEHOLDER_VALUE     = -99.0
IMPORTANT_VERY_IMBALANCED_COLUMNS   = ["NACCFADM", "ELAT", "GAMES", "MOGAIT", "MOSLOW", "BRNINJ", "OTHPSY"]

N_IMPORTANT_FEATURES = 237


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_survival_target(df):
    """Return a structured survival array from *df*."""
    return Surv.from_dataframe(SURVIVAL_EVENT_COL, SURVIVAL_TIME_COL, df)


def split_features_target(df):
    """Drop survival columns and return (X, y)."""
    X = df.drop(columns=[SURVIVAL_TIME_COL, SURVIVAL_EVENT_COL])
    y = build_survival_target(df)
    return X, y

def concat_features_target(X, y):
    """Concatenate features and survival target into a single DataFrame."""
    y_df = pd.DataFrame({
        SURVIVAL_TIME_COL: y[SURVIVAL_TIME_COL],
        SURVIVAL_EVENT_COL: y[SURVIVAL_EVENT_COL]
    })
    return pd.concat([X.reset_index(drop=True), y_df], axis=1)

def keep_available(features, columns):
    """Filter *features* to those present in *columns*."""
    col_set = set(columns)
    return [f for f in features if f in col_set]


def drop_useless_columns(df, columns_to_drop=[]):
    '''
      Drop columns that represent potential leakage variables
      or were considered as useless for the prediction
    '''
    print(f"Dropping useless columns and columns represented the MCI diagnosis")

    df_result = df.copy()
    df_result.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df_result.drop(columns=['NACCACTV', 'NACCADMD', 'NACCALZD', 'NACCALZP', 'PROBAD', 'PROBADIF', 'POSSAD', 'POSSADIF'], inplace=True, errors='ignore')
    df_result.drop(columns=['NACCMCII', 'NACCNORM', 'COGSTAT', 'VISITDAY', 'VISITYR', 'VISITMO', 'NACCETPR'], inplace=True, errors='ignore')
    df_result.drop(columns=['NACCID'], inplace=True, errors='ignore')
    return df_result


def filter_columns_by_missing_pattern(df):
    '''
        Define the missingness pattern by the reference column
        Filter via defined patterns all the columns that are not fully follows it
    '''
    print(f"Filtering columns by missing pattern")

    reference_columns = [
    'DECCLCOG',
    'DECCLBE',
    'DECCLMOT',
    'LBDEVAL',
    'FTLDEVAL',
    'DXMETHOD',
    'OTHBIOM',
    'MSA',
    'FTLDMO',
    'FTLDNOS',
    'CVD',
    'PREVSTK',
    'ESSTREM',
    'EPILEP',
    'HIV',
    'OTHCOG',
    'BIPOLDX',
    'SCHIZOP',
    'ANXIET',
    'DELIR',
    'PTSDDX',
    'IMPSUB',
    'OTHCOND',
    'DECCLIN',
    'WHODIDDX',
    'STROKE'
    ]

    for reference_col in reference_columns:
        if reference_col in df.columns:
            break
    else:
        raise ValueError(f"None of the reference columns {reference_columns} found in the dataframe")

    reference_mask = df[reference_col].isna().to_numpy()

    forward_columns     = []
    opposite_columns    = []

    for col in df.columns:
        col_mask = df[col].isna().to_numpy()

        if np.array_equal(col_mask, reference_mask):
            forward_columns.append(col)
        elif np.array_equal(col_mask, ~reference_mask):
            opposite_columns.append(col)

    kept_columns    = forward_columns + opposite_columns
    dropped_columns = [col for col in df.columns if col not in kept_columns]
    filtered_df     = df[kept_columns].copy()

    return filtered_df, reference_col


def define_missingness(df):
    '''
      Group columns by missingness percentage into low, medium and high missingness groups
    '''
    print(f"Defining missingness")
    missing_percentage_per_column = df.isna().sum() / len(df) * 100

    low_missing     = missing_percentage_per_column[missing_percentage_per_column < LOW_MISSINGNESS_THRESHOLD].index.tolist()
    medium_missing  = missing_percentage_per_column[(missing_percentage_per_column >= LOW_MISSINGNESS_THRESHOLD) & (missing_percentage_per_column < HIGH_MISSINGNESS_THRESHOLD)].index.tolist()
    high_missing    = missing_percentage_per_column[missing_percentage_per_column >= HIGH_MISSINGNESS_THRESHOLD].index.tolist()

    return low_missing, medium_missing, high_missing


def define_categorical_and_continuous_columns(df):
    '''
      Group columns into categorical and continuous based on the number of unique values,
      with a special handling for some unusual categorical columns
    '''
    result_df           = df.copy()
    unusual_categorical = ['NACCBEHF']
    categorical_cols    = []
    continuous_cols     = []
    categorical_cols_by_unique_count = {}

    for col in result_df.columns:
        if col == 'EVENT_MCI':
            continue

        if result_df[col].dtype == 'object':
            n_unique = result_df[col].nunique(dropna=True)
            if n_unique <= 1:
                result_df.drop(columns=[col], inplace=True)
                continue
            categorical_cols.append(col)
            if n_unique not in categorical_cols_by_unique_count:
                categorical_cols_by_unique_count[n_unique] = []
            categorical_cols_by_unique_count[n_unique].append(col)
            continue

        n_unique = pd.to_numeric(result_df[col], errors='coerce').nunique(dropna=True)
        if n_unique == 1:
            result_df.drop(columns=[col], inplace=True)
            continue
        if 2 <= n_unique <= 12 or col in unusual_categorical:
            categorical_cols.append(col)
            if n_unique not in categorical_cols_by_unique_count:
                categorical_cols_by_unique_count[n_unique] = []
            categorical_cols_by_unique_count[n_unique].append(col)
            continue
        continuous_cols.append(col)

    return categorical_cols, continuous_cols, categorical_cols_by_unique_count


def clean_columns(df):
    '''
      Delete columns with very high imbalance (for categorical) or very low variance (for continuous),
      with a special handling for some important but very imbalanced columns
    '''
    print(f"Cleaning columns")
    df_clean = df.copy()
    all_categorical_cols, all_continuous_cols, categorical_cols_by_unique_count = define_categorical_and_continuous_columns(df_clean)

    for col in all_categorical_cols:
        if col not in df_clean.columns:
            continue
        # check 1: overall imbalance
        col_value_proportions = df_clean[col].value_counts(normalize=True, dropna=True)
        if col_value_proportions.iloc[0] > 0.99:
            df_clean.drop(columns=[col], inplace=True)
            continue  # no need to check further

    for col in all_continuous_cols:
        if col not in df_clean.columns:
            continue
        col_var = df_clean[col].var(numeric_only=True)
        if col_var == 0.0000:
            df_clean.drop(columns=[col], inplace=True)
            continue
        if col_var < 0.01 and col not in IMPORTANT_VERY_IMBALANCED_COLUMNS:
            print(f"Column '{col}' has very low variance ({col_var:.6f}); consider to drop it.")
            df_clean.drop(columns=[col], inplace=True)

    # keep column lists synchronized with the actually retained dataframe
    filtered_categorical_cols = keep_available(all_categorical_cols, df_clean.columns)
    filtered_continuous_cols = keep_available(all_continuous_cols, df_clean.columns)

    return df_clean, filtered_categorical_cols, filtered_continuous_cols


def decode_preprocessed_feature_name(feature_name, categorical_cols, continuous_cols):
    '''
      Decode the feature name from the preprocessed format
      (e.g., 'categorical__colname' or 'continuous__colname') to the original raw feature name,
    '''
    if feature_name.startswith('categorical__'):
        remainder = feature_name.replace('categorical__', '', 1)
        matching = [
            col for col in categorical_cols
            if remainder == col or remainder.startswith(f'{col}_')
        ]
        if len(matching) > 0:
            return max(matching, key=len)
        return remainder

    if feature_name.startswith('continuous__'):
        return feature_name.replace('continuous__', '', 1)

    return feature_name


def select_features_subset(df, feature_names, categorical_cols, continuous_cols):
    '''
      After feature selection select the subset of features
      by decoding each of them and finding in initial dataset the corresponding raw feature name
    '''
    raw_features = []
    for feature in feature_names:
        base_name = decode_preprocessed_feature_name(feature, categorical_cols, continuous_cols)

        if base_name in df.columns and base_name not in raw_features:
            raw_features.append(base_name)

    for required_col in ['TIME', 'EVENT_MCI', "group_missing_indicator"]:
        if required_col in df.columns and required_col not in raw_features:
            raw_features.append(required_col)

    return df[raw_features]


def low_missingness_complete_case_analysis(df, low_missingness_columns=None):
    '''
       Perform complete-case analysis on low-missing columnsm
       Default: define missingness columns itself
       if low missingness columns procided, perform complete-case analysis on them
    '''
    print(f"Complete-case analysis on low-missing columns")
    df_result = df.copy()

    if low_missingness_columns is None:
        low_missingness_columns = define_missingness(df_result)[0]

    if len(low_missingness_columns) > 0:
        nacc_missing_free_v2_v3 = df_result.dropna(
            subset=low_missingness_columns
        ).copy()
    else:
        nacc_missing_free_v2_v3 = df_result.copy()
    return nacc_missing_free_v2_v3


def create_missingness_indicators(df, column_ref_indicator='HIV'):
    print(f"Creating missingness indicator")
    df_result = df.copy()
    df_result['group_missing_indicator'] = np.where(
        df_result[column_ref_indicator].isna(), 1, 0
    )
    return df_result


# ---------------------------------------------------------------------------
# Pipeline custom components
# ---------------------------------------------------------------------------

class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05, categorical_cols=None, non_collected_placeholder=None):
        self.threshold                  = threshold
        self.categorical_cols           = categorical_cols
        self.non_collected_placeholder  = non_collected_placeholder
        self.rare_categories_           = {}

    def fit(self, X, y=None):
        if self.non_collected_placeholder is None:
            raise ValueError('non_collected_placeholder cannot be None')

        df_result = X.copy()
        self.feature_names_in_ = np.asarray(df_result.columns, dtype=object)
        for col in self.categorical_cols:
            if col not in df_result.columns:
                continue

            value_counts                = df_result[col][df_result[col] != self.non_collected_placeholder].value_counts(normalize=True)
            rare_categories             = value_counts[value_counts < self.threshold].index.tolist()
            self.rare_categories_[col]  = rare_categories

        return self

    def transform(self, X):
        df_result = X.copy()

        for col, rare_categories in self.rare_categories_.items():
            if col in df_result.columns and len(rare_categories) > 0:
                rare_categories_str   = ', '.join(map(str, rare_categories))
                df_result[col]        = df_result[col].replace(rare_categories, f'{col}_{rare_categories_str}')
        return df_result

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols=None, selected_features_subset=None):
        self.categorical_cols           = categorical_cols
        self.selected_features_subset   = selected_features_subset
        self.encoder                    = None

    def fit(self, X, y=None):
        df_result       = X.copy()
        categorical_df  = df_result[self.categorical_cols].astype(str)
        self.encoder    = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(categorical_df)

        feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
        self.keep_columns_indices_ = [
            i for i, name in enumerate(feature_names)
            if not name.endswith(f'_{NOT_COLLECTED_PLACEHOLDER_VALUE}') and not name.endswith(f'_{NOT_COLLECTED_PLACEHOLDER_VALUE:.1f}')
        ]
        self.feature_names_out_ = feature_names[self.keep_columns_indices_]

        return self

    def transform(self, X):
        df_result       = X.copy()
        categorical_df  = df_result[self.categorical_cols].astype(str)
        encoded_array   = self.encoder.transform(categorical_df)
        encoded_array   = encoded_array[:, self.keep_columns_indices_]
        return encoded_array

    def get_feature_names_out(self):
        return np.asarray(self.feature_names_out_, dtype=object)


class CustomConstantImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value=None):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        if self.fill_value is None:
            raise ValueError('fill_value cannot be None')
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        df_result = X.copy()
        df_result.fillna(self.fill_value, inplace=True)
        return df_result

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


# ---------------------------------------------------------------------------
# Preprocessing pipeline builder
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(categorical_columns, continuous_columns, selected_features_subset=None):
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', CustomConstantImputer(fill_value=NOT_COLLECTED_PLACEHOLDER_VALUE)),
        ('rare_collapser', RareCategoryCollapser(
            categorical_cols=categorical_columns,
            non_collected_placeholder=NOT_COLLECTED_PLACEHOLDER_VALUE,
        )),
        ('encoder', CustomOneHotEncoder(categorical_cols=categorical_columns, selected_features_subset=selected_features_subset)),
    ]).set_output(transform='pandas')

    # Continuous features pipeline
    continious_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('boxcox-transformer', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler()),
    ]).set_output(transform='pandas')

    columns_preprocessing_pipeline = ColumnTransformer([
        ('categorical', categorical_pipeline, categorical_columns),
        ('continuous', continious_pipeline, continuous_columns),
    ]).set_output(transform='pandas')

    return Pipeline([
        ('columns_preprocessing', columns_preprocessing_pipeline),
    ]).set_output(transform='pandas')


# ---------------------------------------------------------------------------
# Dataset preprocessor
# ---------------------------------------------------------------------------

class BaseDatasetPreprocessor:
    def __init__(self):
        self._cleaned_df          = None
        self._low_missing_cols    = None
        self._categorical_cols    = None
        self._continuous_cols     = None
        self._pipeline            = None
        self._pipeline_input_cols = None
        self._retained_columns    = None

    def fit(self, X, y=None):
        self._fit_structural_cleanup(X)
        self._fit_transform_pipeline(self._get_features_data())
        return self

    def transform(self, X, y=None):
        self._assert_pipeline_fitted()
        self._apply_structural_cleanup(X)
        return self._transform_pipeline(self._get_features_data())

    def fit_transform(self, X, y=None):
        self._fit_structural_cleanup(X)
        return self._fit_transform_pipeline(self._get_features_data())

    def _get_features_data(self):
        features_data = [c for c in self._cleaned_df.columns
                        if c not in (SURVIVAL_EVENT_COL, SURVIVAL_TIME_COL)]
        return self._cleaned_df[features_data]

    def _fit_structural_cleanup(self, X):
        """
          Data-dependent cleaning: drop useless columns, apply missingness filtering,
          create missingness indicators, infer column types, and drop low-variance /
          imbalanced columns. Stores the resulting column structure for transform.
        """
        X_copy = drop_useless_columns(X.copy())

        low_missing, _medium_missing, _high_missing = define_missingness(X_copy)
        medium_filtered, self.pattern_ref_col = filter_columns_by_missing_pattern(X_copy[_medium_missing])

        retained_cols   = medium_filtered.columns.tolist() + low_missing
        X               = X[retained_cols].copy()

        X = create_missingness_indicators(X, column_ref_indicator=self.pattern_ref_col)
        X, categorical_cols, continuous_cols = clean_columns(X)

        self._categorical_cols = [c for c in categorical_cols if c not in (SURVIVAL_EVENT_COL, SURVIVAL_TIME_COL)]
        self._continuous_cols  = [c for c in continuous_cols  if c not in (SURVIVAL_EVENT_COL, SURVIVAL_TIME_COL)]
        self._cleaned_df       = X
        self._low_missing_cols = low_missing
        self._retained_columns = list(X.columns)

    def _apply_structural_cleanup(self, df):
        """
          Apply the column structure learned during fit to new data.
          Does not recompute any data-dependent decisions.
        """
        df = drop_useless_columns(df.copy())
        df = create_missingness_indicators(df, column_ref_indicator=self.pattern_ref_col)
        available_cols = [c for c in self._retained_columns if c in df.columns]
        self._cleaned_df = df[available_cols].copy()

    def _fit_transform_pipeline(self, X):
        actual_categoricals  = keep_available(self._categorical_cols, X.columns)
        actual_continuous = keep_available(self._continuous_cols,  X.columns)

        self._pipeline            = build_preprocessing_pipeline(actual_categoricals, actual_continuous)
        self._pipeline_input_cols = list(X.columns)
        return self._pipeline.fit_transform(X)
    
    def _transform_pipeline(self, X):
        missing_cols = set(self._pipeline_input_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        X_aligned = X.reindex(columns=self._pipeline_input_cols)
        return self._pipeline.transform(X_aligned)
    
    def get_low_missingness_cols(self):
        """Return the low-missingness columns identified during fit."""
        if self._low_missing_cols is None:
            raise RuntimeError(
                "Preprocessor is not fitted. Call fit() or fit_transform() first."
            )
        return list(self._low_missing_cols)

    def _assert_pipeline_fitted(self):
        if self._pipeline is None:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit_transform() first."
            )
