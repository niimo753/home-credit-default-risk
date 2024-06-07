# LIBRARIES
# Pandas for data manipulation and analysis
import pandas as pd
import numpy as np

# Check if the dtype of a column is object
from pandas.api.types import is_object_dtype

# Imputation for handling missing values in the dataset
from sklearn.impute import SimpleImputer

# For encoding categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder, OrdinalEncoder

# For scaling and transforming numerical features
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer

# For enerating polynomial features
from sklearn.preprocessing import PolynomialFeatures

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.linear_model import LogisticRegression


# FUNCTIONS
# Data Exploration
def replace_nan(data, missing_value=['XNA', 'Unknown', 365243]):
    '''
    Replace all missing values that have not been defined by np.nan with np.nan
    
    Parameters:
    - data (pandas.DataFrame): data having values needed to be replaced
    - missing_value (list): A list of all missing values but have not been defined by np.nan, default=['XNA', 'Unknown', 365243]
    
    Return:
    pandas.DataFrame: A dataframe with all missing values (except np.nan) having been redefined by np.nan
    '''
    for value in missing_value:
        data = data.replace(value, np.nan)
    return data
    
def data_explore(data):
    '''
    General information about the dataset (number of rows, columns, duplicated rows and number of features corresponding to each type)
    
    Parameters:
    data (pandas.DataFrame): A dataframe needed to be explored
        
    Return:
    pandas.DataFrame: A dataframe with basic information about the input dataframe
    
    '''
    # Number of rows, columns and duplicated rows
    info = [['Rows', data.shape[0]],
            ['Features', data.shape[1]],
            ['Duplicate Rows', data.duplicated().sum()]]
    datainfo = pd.DataFrame(info, columns=['index', 'info']).set_index('index')
    
    # Number of features corresponding to each type
    dtype = pd.DataFrame(data.dtypes.value_counts()).rename(columns={'count': 'info'})
    
    # Final result
    datainfo = pd.concat([datainfo, dtype], axis=0)
    
    return datainfo

# Check Missing Value
def check_nan(data, axis=1):
    '''
    Return the number and percentage of NaN value corresponding to chosen axis in dataset
    
    Parameters:
    - data (pandas.DataFrame): Data that contains features you want to explore
    - axis (bool/int) {True, False, 1, 0}: default=1
        + If True or 1, return missing value per column
        + If False or 0, return missing value per row
        
    Return:
    pandas.DataFrame: A dataframe contains information about the number and percentage of NaN values corresponding to chosen axis
    
    '''
    output = pd.DataFrame(data.isnull().sum(1-axis), columns=['nan']).sort_values(by='nan', ascending=False)
    output = output[output['nan']>0]
    output['%nan'] = output['nan']/data.shape[1-axis]*100
    return output
    

# All Feature Exploration
def multi_features_explore(data, reverse=False):
    '''
    Explore multiple features of the dataset
    
    Parameters:
    - data (pandas.DataFrame): Data that contains features you want to explore
    - reverse (bool): default=False
        + If False, output will be a DataFrame with feature's name as columns
        + If True, output will be a DataFrame with feature's name as indexes
        
    Return:
    pandas.DataFrame: A dataframe of basic information about all features of the input data
    
    '''
    colinfo = []
    for col in data:
        colinfo.append(feature_explore(data[col]))
    if not reverse:
        return pd.concat(colinfo, axis=1).T
    else:
        return pd.concat(colinfo, axis=1)
    
# Single Feature Exploration
def feature_explore(feature):
    '''
    General information of a specific feature
    
    Parameters:
    feature (pandas.Series): Feature needed to be explored
    
    Return:
    pandas.DataFrame:  A dataframe of basic information about a specific feature
    
    '''
    info = [['dtype', feature.dtype],
            ['nonnull', feature.notnull().sum()],
            ['%nonnull', round(feature.notnull().sum()/feature.shape[0], 2)],
            ['nan', feature.isnull().sum()],
            ['%nan', round(feature.isnull().sum()/feature.shape[0], 2)],
            ['nunique', feature.nunique()],
            ['nunique_nan', len(feature.unique())]]
    if is_object_dtype(feature):
        info.extend([['unique', feature.unique()],
                     ['frequency', feature.value_counts().to_dict()],
                     ['%value', feature.value_counts(normalize=True).round(2).to_dict()],
                     ['most', feature.mode().values]])
    else:
        if feature.nunique() <= 10:
            info.extend([['unique', feature.unique()],
                         ['frequency', feature.value_counts().to_dict()],
                         ['%value', feature.value_counts(normalize=True).round(2).to_dict()]])
        info.extend([['max', feature.max()],
                     ['min', feature.min()],
                     ['mean', feature.mean()],
                     ['std', feature.std()]])
    output = pd.DataFrame(info, columns=['index', feature.name])
    output = output.set_index('index')
    return pd.DataFrame(output)

# Distribution Of Numerical Features Using Histplot
def num_dist_histplot(data, nrows=1, ncols=1, figsize=(5,5), fontsize=10, bins=30, limit=-1):
    '''
    Using histplot to show the general distribution of at least one numerical feature
    
    Parameters:
    - data (pandas.DataFrame): A dataframe with all numerical features needed to be plot
    - nrows (int) Number of rows of subplots,  default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appear in the plot, default=10 
    - bins (int): Number of bins, default=30
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots.
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    cols = list(data.columns)
    index = 0
    if nrows == 1 and ncols == 1:
        sns.histplot(data[cols[index]].sort_values(), bins=bins, ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel=cols[index], fontsize=fontsize)
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            i.tick_params(labelsize=fontsize)
            sns.histplot(data[cols[index]].sort_values(), bins=bins, ax=i)
            i.set_ylabel("")
            i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                i.tick_params(labelsize=fontsize)
                sns.histplot(data[cols[index]].sort_values(), bins=bins, ax=i)
                i.set_ylabel("")
                i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
                index += 1
                if index == limit:
                    break

# Distribution Of Numerical Features Using Boxplot                    
def num_dist_boxplot(data, nrows=1, ncols=1, figsize=(5,5), fontsize=10, limit=-1):
    '''
    Using boxplot to show the general distribution of at least one numerical feature, especially information of the outliers
    
    Parameters:
    - data (pandas.DataFrame): A dataframe with all numerical features needed to be plot
    - nrows (int) Number of rows of subplots,  default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appear in the plot, default=10 
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots.
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    cols = list(data.columns)
    index = 0
    if nrows == 1 and ncols == 1:
        data.boxplot(column=cols[index], ax=ax, fontsize=fontsize)
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            data.boxplot(column=cols[index], ax=i, fontsize=fontsize)
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                data.boxplot(column=cols[index], ax=i, fontsize=fontsize)
                index += 1
                if index == limit:
                    break
                
def num_kdeplot(data, cols, color, nrows=1, ncols=1, figsize=(5, 5), fontsize=10, limit=-1):
    '''
    Visualize the effect of some numerical features on the target using kdeplot.
    
    Parameters
    - data (pandas.DataFrame): A dataframe with all numerical features needed to be visualized
    - cols (list): A list of features name needed to be visualized
    - colors (str): Name of the feature that will be used as a categorical separation of data, allowing you to color the chart based on its unique values.
    - nrows (int): Number of rows of subplots, default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appeared in the plot, default=10
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots.  
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    index = 0
    if nrows == 1 and ncols == 1:
        for val in data[color].unique():
            sns.kdeplot(data.loc[data[color] == val, cols[index]], label = f'{color} = {val}', ax=ax, common_norm=False)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel=cols[index], fontsize=fontsize)
        ax.legend(fontsize=fontsize)
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            for val in data[color].unique():
                sns.kdeplot(data.loc[data[color] == val, cols[index]], label = f'{color} = {val}', ax=i, common_norm=False)
            i.tick_params(labelsize=fontsize)
            i.set_ylabel("")
            i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
            i.legend(fontsize=fontsize)
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                for val in data[color].unique():
                    sns.kdeplot(data.loc[data[color] == val, cols[index]], label = f'{color} = {val}', ax=i, common_norm=False)
                i.tick_params(labelsize=fontsize)
                i.set_ylabel("")
                i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
                i.legend(fontsize=fontsize)
                index += 1
                if index == limit:
                    break
                    
def dist_base_barplot(data, cols, color, nrows=1, ncols=1, figsize=(5, 5), fontsize=10, limit=-1, norm=True):
    '''
    Visualize the distribution of some features based on the distribution of a specific features uisng barplot.
    
    Parameters
    - data (pandas.DataFrame): A dataframe with all features needed to be plot
    - col (str): A feature will appear on x-axis
    - colors (list): A list of features name that are used as a categorical separation of data, allowing you to color the bars based on their unique values.
    - nrows (int): Number of rows of subplots, default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appeared in the plot, default=10
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots. 
    - norm (bool): default=True
        + If True, normalize the data before visualizing.
        + If False, using the original data to visualize.
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    index = 0
    if nrows == 1 and ncols == 1:
        temp = data.groupby(by=cols, as_index=False)[color[index]].value_counts(normalize=norm)
        sns.barplot(data=temp, x=cols, y=temp.columns[-1], hue=color[index], ax=ax)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel=color[index], fontsize=fontsize)
        ax.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            temp = data.groupby(by=cols, as_index=False)[color[index]].value_counts(normalize=norm)
            sns.barplot(data=temp, x=cols, y=temp.columns[-1], hue=color[index], ax=i)
            i.tick_params(labelsize=fontsize)
            i.set_ylabel("")
            i.set_xlabel(xlabel=color[index], fontsize=fontsize)
            if data[color[index]].nunique() <= 10:
                i.legend(fontsize=fontsize, bbox_to_anchor=(1, 1), loc='upper left')
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                temp = data.groupby(by=cols, as_index=False)[color[index]].value_counts(normalize=norm)
                sns.barplot(data=temp, x=cols, y=temp.columns[-1], hue=color[index], ax=i)
                i.tick_params(labelsize=fontsize)
                i.set_ylabel("")
                i.set_xlabel(xlabel=color[index], fontsize=fontsize)
                if data[color[index]].nunique() <= 10:
                    i.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))
                index += 1
                if index == limit:
                    break

def poly(data, features, degree=3):
    '''
    Generate polynomial and interaction features. Returns the original dataframe plus all newly created features.
    
    Parameters:
    - data (pandas.DataFrame): a dataframe that contains features that needed to be generated
    - features (list): a list contains all name of features that needed to be generated
    - degree (int or tuple): If a single int is given, it specifies the maximal degree of the polynomial features. If a tuple (min_degree, max_degree) is passed, then min_degree is the minimum and max_degree is the maximum polynomial degree of the generated features. Default: 2
    
    '''
    # separate train and test set
    train, test = data[data['TARGET'].notnull()], data[data['TARGET'].isnull()].reset_index()
    
    # the output of PolynomialFeatures will contains 1 column of all 1s and n columns of n original features, which are not neccessary, we will use basenum to remove it later
    basenum = len(features)+1
    
    # define features need to be generated
    poly_features_train = train[features].copy()
    poly_features_test = test[features].copy()
    
    # generate polynomial and interaction features
    polyfea = PolynomialFeatures(degree=degree)
    polyfea.fit(poly_features_train)
    poly_features_train = polyfea.transform(poly_features_train)
    poly_features_test = polyfea.transform(poly_features_test)
    
    # add newly created features to the original data
    poly_features_train_df = pd.DataFrame(poly_features_train, columns = polyfea.get_feature_names_out())
    poly_features_test_df = pd.DataFrame(poly_features_test, columns = polyfea.get_feature_names_out())
    poly_features_train_df['SK_ID_CURR'] = train['SK_ID_CURR']
    poly_features_test_df['SK_ID_CURR'] = test['SK_ID_CURR']
    poly_features_df = pd.concat([poly_features_train_df, poly_features_test_df], axis=0, ignore_index=True)
    poly_features_df = poly_features_df.iloc[:, basenum:].copy()
    
    result = data.merge(poly_features_df, on="SK_ID_CURR", how='left')
    
    return result

# General Encoding Function
def general_encode(data, features, method='one-hot', train_test_separate=False, categories=None):
    '''
    Encodes categorical columns in the given DataFrame based on the specified method.

    Parameters:
    - data (pandas DataFrame): The input DataFrame containing the categorical columns.
    - features (list): List of features to be encoded.
    - method (str, optional): default="one-hot", the encoding method. Options: "one-hot", "label", "ordinal", "target".
    - train_test_separate (bool, optional): default=True, often used for "target" method.
        + If True, separate the input dataframe into two parts: train set and test set. First, fit and transform the train set, then transform the test set. Must be used for "target" method
        + If False, fit and transform the whole input dataframe.
    - categories (None or a list of array-like): default=None, specific parameter for "ordinal" method. 
        + None: no specific categories required or method used is not "ordinal"
        + list : categories[i] holds the categories expected in the ith column. The passed categories should not mix strings and numeric values, and should be sorted in case of numeric values.

    Returns:
    - pandas DataFrame
        The DataFrame with encoded categorical columns.
        
    '''
    if train_test_separate:
        encoded_data, test_data = data[data['TARGET'].notnull()].copy(), data[data['TARGET'].isnull()].copy()
    else:
        encoded_data, test_data = data.copy(), None
        
    if method == 'one-hot':
        encoded_data = pd.get_dummies(encoded_data, columns=features, dtype=int)
        if test_data is not None and test_data.shape[0]>0:
            test_data = pd.get_dummies(test_data, columns=features, dtype=int)

    elif method == 'label':
        for column in features:
            label_encoder = LabelEncoder()
            encoded_data[column] = label_encoder.fit_transform(encoded_data[column])
            if test_data is not None and test_data.shape[0]>0:
                test_data = label_encoder.transform(test_data[column])

    elif method == 'ordinal':
        if not categories:
            categories = "auto"
        for column in features:
            ordinal_encoder = OrdinalEncoder(categories=categories)
            encoded_data[column] = ordinal_encoder.fit_transform(encoded_data[[column]])
            if test_data is not None and test_data.shape[0]>0:
                test_data = ordinal_encoder.transform(test_data[[column]])

    elif method == 'target':
        for column in features:
            target_encoder = TargetEncoder()
            encoded_data[column] = target_encoder.fit_transform(encoded_data[[column]], encoded_data[['TARGET']])
            if test_data is not None and test_data.shape[0]>0:
                test_data[column] = target_encoder.transform(test_data[[column]])
            else:
                raise ValueError("Check if the test set included and remember to set the train_test_separate to True")

    else:
        raise ValueError("Invalid encoding method. Choose from 'one-hot', 'label', 'ordinal', 'target'.")
    
    if test_data is not None and test_data.shape[0]>0:
        encoded_data = pd.concat([encoded_data, test_data])

    return encoded_data

# Specific Encoding Function
def specific_encode(data, basic_mode, custom=None, categories=None):
    '''
    Custom encoding for the whole dataframe.
    
    Parameters:
    - data (pandas.DataFrame): The input DataFrame containing the categorical columns.
    - basic_mode (bool): default=True
        + If True, using LabelEncoder with categorical columns having less than 2 unique values and OneHotEncoder for the rest.
        + If False, using "custom" parameters to defined your own encoding style
    - custom (None, str or dict): default=None,  must be used if basic_mode is False
        + str: Name of the encode method that will be used for all categorical data
        + dict: a dictionary for your own encoding style with each key is a method and value is a list of columns will be encoded by that method.
    - categories (None or a list of array-like): default=None, specific parameter for "ordinal" method. 
        + None: no specific categories required or method used is not "ordinal"
        + list : categories[i] holds the categories expected in the ith column. The passed categories should not mix strings and numeric values, and should be sorted in case of numeric values.
        
    Returns:
    pandas.DataFrame: The DataFrame with encoded categorical columns.
        
    '''
    encoded_data = data.copy()
    
    if basic_mode:
        encoded_features_name = data.select_dtypes("object").columns
        for feature in encoded_features_name:
            if data[feature].nunique() <= 2:
                encoded_data = general_encode(encoded_data, features=[feature], method='label')
            else:
                encoded_data = general_encode(encoded_data, features=[feature], method='one-hot')
    
    else:
        if isinstance(custom, str):
            columns = list(encoded_data.select_dtypes('object').columns)
            if custom=='target':
                encoded_data = general_encode(encoded_data, method=custom, features=columns, train_test_separate=True)
            elif custom=='ordinal':
                encoded_data = general_encode(encoded_data, method=custom, features=columns, categories=categories)
            else:
                encoded_data = general_encode(encoded_data, method=custom, features=columns)
        else:
            for key, value in custom.items:
                if key == 'ordinal':
                    encoded_data = general_encode(encoded_data, method=key, features=value, categories=categories)
                elif key == 'target':
                    encoded_data = general_encode(encoded_data, method=key, features=value, train_test_separate=True)
                else:
                    encoded_data = general_encode(encoded_data, method=key, features=value)
        
    return encoded_data

# Handling categorical features in subtables
def sub_cate_norm(df, groupbycol, prefix):
    """
    Normalize categorical data and calculate summary statistics.
    
    First creates dummy variables for the categorical 
    columns in the input dataframe, calculates the 'sum' and 'mean' of each categorical column for 
    each group after groupby and returns the resulting dataframe.
    
    Parameters:
    
    - df (pandas.DataFrame): Input dataframe.
    - groupbycol (str): Column to group the dataframe by.
    - prefix (str): Prefix to add to the column names in the output 
    dataframe.
    
    Returns:
    
    catedf (pandas.DataFrame): Normalized categorical data and 
    summary statistics.
    """
    cate = pd.get_dummies(df.select_dtypes('object'))
    cate[groupbycol] = df[groupbycol]
    catedf = cate.groupby(groupbycol).agg(['sum','mean'])
    cols = []
    for i in catedf.columns.levels[0]:
        for j in ['count','norm']:
            cols.append('%s_%s_%s' % (prefix,i,j))
    catedf.columns = cols
    return catedf

# Handling numerical features in subtables
def sub_num_agg(df, groupbycol, prefix):
    '''
    Aggregate numerical data and calculate summary statistics.
    
    First removes all columns from the input dataframe that are not numerical or do not have the specified 'groupbycol', calculates 'mean', 'max', 'min' and 'sum' of each numerical column for each group after groupby and returns the resulting dataframe.
    
    Parameters:
    - df (pandas.DataFrame): Input dataframe.
    - groupbycol (str): Column to group the dataframe by.
    - prefix (str): Prefix to add to the column names in the output 
    dataframe.
    
    Returns:
    numdf (pandas.DataFrame): Aggregated numerical data and summary statistics.
    '''
    for col in df:
        if col != groupbycol and 'SK_ID' in col:
            df = df.drop(columns=col)
    id_col = df[groupbycol]
    num_df = df.select_dtypes('number')
    num_df[groupbycol] = id_col

    agg = num_df.groupby(groupbycol).agg(['min', 'max', 'mean', 'sum']).reset_index()
    cols = [groupbycol]
    for i in agg.columns.levels[0]:
        if i != groupbycol:
            for j in agg.columns.levels[1][:-1]:
                cols.append('%s_%s_%s' % (prefix, i, j))
    agg.columns = cols
    return agg

def scale_df_full(df, method='standard'):
    '''
    Scale numerical data in a dataframe.
    
    This function scales the numerical data in the input dataframe using the min-max scaling method or the standard scaling method. It returns the original dataframe with the numerical columns replaced by their respective scaled versions.
    
    Parameters:
    - df (pandas.DataFrame): Input dataframe.
    - method (str): Scaling method. Can be 'minmax' for min-max scaling or 'standard' for standard scaling.
    
    Returns:
    scaled_df (pandas.DataFrame): Dataframe with scaled numerical data.
    '''
    # Determine which columns are numeric
    numeric_cols = df.select_dtypes(exclude='object').columns

    # Initialize a dictionary to store the scaling instances
    scalers = {}

    # Choose the appropriate scaling method
    if method == 'minmax':
        Scaler = MinMaxScaler
    elif method == 'standard':
        Scaler = StandardScaler
    else:
        raise ValueError("Invalid scaling method. Must be 'minmax' or 'standard'.")

    # Fit the scaling instances and transform the numeric columns
    for col in numeric_cols:
        if col not in scalers:
            scalers[col] = Scaler()
        df[col] = scalers[col].fit_transform(df[[col]])

    return df

def scale_df_columns(df, min_max_cols=None, standard_cols=None):
    '''
    Scale numerical data in a dataframe.
    
    This function scales the numerical data in the input dataframe using the min-max scaling method or the standard scaling method. It returns the original dataframe with the numerical columns replaced by their respective scaled versions.
    
    Parameters:
    - df (pandas.DataFrame): Input dataframe.
    - min_max_cols (list): List of column names to be scaled using the min-max scaling method. If None, no columns will be scaled using this method.
    - standard_cols (list): List of column names to be scaled using the standard scaling method. If None, no columns will be scaled using this method.
    
    Returns:
    scaled_df (pandas.DataFrame): Dataframe with scaled numerical data.
    '''
    scaled_df = df.copy()

    if min_max_cols is not None:
        scaler = MinMaxScaler()
        scaled_df[min_max_cols] = scaler.fit_transform(df[min_max_cols])

    if standard_cols is not None:
        scaler = StandardScaler()
        scaled_df[standard_cols] = scaler.fit_transform(df[standard_cols])

    return scaled_df

def continuous_plot(data, col, plot=['dist', 'box'], figsize=(20, 8)):
    '''
    Creates continuous plots for a specified column in a given DataFrame.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - col (str): The column in the DataFrame for which the plots are to be created.
    - plot (list of str): The type of plots to be created. It should be a list of either 'dist' or 'box'.
    - figsize (tuple of int): The size of the figure to be displayed.
    
    Returns:
    None: This function does not return any value.
    '''
    num_sub = len(plot)
    fig, ax = plt.subplots(1, num_sub, figsize=figsize)
    
    for i, p in enumerate(plot):
        if p == 'dist':
            # Plotting kernel density estimate for both target classes
            sns.kdeplot(data.loc[data['TARGET'] == 0, col], label='0', ax=ax[i])
            sns.kdeplot(data.loc[data['TARGET'] == 1, col], label='1', ax=ax[i])
            
            # Setting legend, xlabel, and ylabel
            ax[i].legend()
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('Density')
        elif p == 'box':
            # Plotting boxplot for the specified column and target classes
            sns.boxplot(x=data['TARGET'], y=data[col], ax=ax[i])

def check_corr(data, heatmap=True, figsize=(20, 20)):
    '''
    Checks the correlation between features in a DataFrame.

    Parameters:
    - data (DataFrame): The input DataFrame containing numerical features.
    - heatmap (bool, optional): default=True
        + If True, displays a heatmap of the correlation matrix.
        + If False, returns the correlation matrix.

    Returns:
    DataFrame or None: Returns the correlation matrix if heatmap is False, otherwise, None.
    '''
    # Check if heatmap argument is True to create and display the heatmap
    if heatmap:
        # Create a mask for the upper triangle of the heatmap
        mask = np.triu(np.ones_like(data.corr()))

        # Create the heatmap figure
        plt.figure(figsize=figsize)

        # Generate the heatmap using seaborn, annotating the cells with correlation values
        sns.heatmap(data.corr(), cmap="YlGnBu", annot=True, mask=mask)

    # If heatmap argument is False, return the correlation matrix
    else:
        return data.corr()

# Filling Missing Values
def fillna(data, strategy, fill_value=None):
    '''
    Fill missing values in the input DataFrame using SimpleImputer.

    Parameters:
    - data (DataFrame): The input dataframe with missing values.
    - strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', or 'constant')
        + If "mean", then replace missing values using the mean along each column. Can only be used with numeric data.
        + If "median", then replace missing values using the median along each column. Can only be used with numeric data.
        + If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
        + If "constant", then replace missing values with fill_value. Can be used with strings or numeric data.
    - fill_value (str or numerical value): default=None. When strategy is "constant", fill_value is used to replace all occurrences of missing_values. For string or object data types, fill_value must be a string. If None, fill_value will be 0 when imputing numerical data and "missing_value" for strings or object data types.

    Returns:
    DataFrame: The dataframe with missing values imputed using the specified strategy.

    '''
    imp = SimpleImputer(missing_values=np.nan, strategy= strategy)
    df = pd.DataFrame(imp.fit_transform(data), columns = data.columns).astype(data.dtypes.to_dict())
    return df

def fillna_occupation(data):
    '''
    Fills missing 'OCCUPATION_TYPE' values based on related categorical columns.

    Parameters:
    data (DataFrame): Input DataFrame containing various categorical and numerical columns,
                       including 'OCCUPATION_TYPE' with missing values. Can use this for test data.

    Returns:
    DataFrame: DataFrame with missing 'OCCUPATION_TYPE' values filled.
    '''
    # Create a copy to avoid modifying the original data
    train = data.copy()

    # Selecting only columns of object type (categorical columns)
    train_cat = train.select_dtypes(include='object')
    train_cat["SK_ID_CURR"] = train["SK_ID_CURR"]

    # Adding 'AMT_INCOME_TOTAL' column to the categorical columns
    train_cat['AMT_INCOME_TOTAL'] = train['AMT_INCOME_TOTAL']

    # Grouping by specific categorical columns along with 'OCCUPATION_TYPE'
    # Calculating the mean 'AMT_INCOME_TOTAL' for each group
    criteria_occupation = train_cat.groupby(['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index()

    # Selecting rows where 'OCCUPATION_TYPE' is NaN (missing)
    train_cat_null = train_cat[train_cat['OCCUPATION_TYPE'].isna()]

    # Columns used for merging
    columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

    # Merging null values with criteria_occupation to fill missing 'OCCUPATION_TYPE'
    train_cat_null1 = pd.merge(train_cat_null, criteria_occupation, how='left', on=columns)

    # Dropping the original NaN 'OCCUPATION_TYPE' column
    train_cat_null1.drop(columns='OCCUPATION_TYPE_x', inplace=True)

    # Renaming the new 'OCCUPATION_TYPE' column
    train_cat_null1.rename(columns={'OCCUPATION_TYPE_y': 'OCCUPATION_TYPE'}, inplace=True)

    # Calculating income total gap between original and filled 'AMT_INCOME_TOTAL'
    train_cat_null1['AMT_INCOME_TOTAL_GAP'] = abs(train_cat_null1['AMT_INCOME_TOTAL_y'] - train_cat_null1['AMT_INCOME_TOTAL_x'])

    # Sorting by income total gap
    train_cat_null1 = train_cat_null1.sort_values('AMT_INCOME_TOTAL_GAP')

    # Keeping the first unique 'SK_ID_CURR' after sorting
    train_cat_null1.drop_duplicates('SK_ID_CURR', keep='first', inplace=True)
    train_cat_null1 = train_cat_null1[['SK_ID_CURR', 'OCCUPATION_TYPE']].copy()
    
    
    train_cat = train_cat.merge(train_cat_null1, on='SK_ID_CURR', how='left')
    train_cat['OCCUPATION_TYPE'] = train_cat[['OCCUPATION_TYPE_x', 'OCCUPATION_TYPE_y']].apply(lambda x: x['OCCUPATION_TYPE_x'] if pd.notna(x['OCCUPATION_TYPE_x']) else x['OCCUPATION_TYPE_y'], axis=1)
    train_cat = train_cat.drop(['OCCUPATION_TYPE_x', 'OCCUPATION_TYPE_y'], axis=1)
    
    # Filling remaining missing 'OCCUPATION_TYPE' values using 'fillna' function (defined before) using 'most_frequent' strategy
    train_cat[['OCCUPATION_TYPE']] = fillna(train_cat[['OCCUPATION_TYPE']], 'most_frequent')

    
    result = train.merge(train_cat[['SK_ID_CURR', 'OCCUPATION_TYPE']], on='SK_ID_CURR', how='left', suffixes=('_OLD', ''))
    result = result.drop('OCCUPATION_TYPE_OLD', axis=1)
    return result

def power_transform(data):
    '''
    Applies PowerTransformer to numerical columns in the DataFrame.

    Parameters:
    data (DataFrame): Input DataFrame containing numerical and possibly categorical columns.

    Returns:
    DataFrame: Transformed DataFrame with power-transformed numerical columns.
    '''
    # Select only the numerical columns from the DataFrame
    data_num = data.select_dtypes(exclude='object')

    # Initialize PowerTransformer
    pt = PowerTransformer()

    # Fit and transform the numerical data using PowerTransformer
    transformed_data = pt.fit_transform(data_num)

    # Convert the transformed data back to a DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=data_num.columns)

    return transformed_df

# Tuning parameters
def objective(trial, X, y):
    # Define the search space for hyperparameters
    C = trial.suggest_float('C', 0.1, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    tol = trial.suggest_float('tol', 1e-5, 1e-3, log=True)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Logistic Regression with Optuna suggested hyperparameters
    log_reg = LogisticRegression(C=C, max_iter=max_iter, penalty= 'l2', solver=solver, tol=tol)

    # Fit Logistic Regression on the training data
    log_reg.fit(X_train, y_train)

    # Calculate validation accuracy
    val_accuracy = log_reg.score(X_val_scaled, y_val)

    return val_accuracy

def tuning_with_optuna(train_data):
    '''
    Optimize Logistic Regression hyperparameters using Optuna.

    Parameters:
    train_data (DataFrame): training data

    Returns:
    dict: A dictionary containing the best hyperparameters found by Optuna.
    '''
    X = train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y = train_data['TARGET']

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    return best_params