import pandas as pd

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    
    # Drop 'Id' column and columns with too many missing values
    data = data.drop(columns=['Id'])
    data.drop(["PoolQC", "Fence", "Alley", "MiscFeature"], axis=1, inplace=True)

    # Impute missing values with 'NA'
    impute_with_na = ["MasVnrType", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                      "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    
    for col in impute_with_na:
        data[col] = data[col].fillna(value="NA")
    
    # Replace missing values in 'Electrical' with the most frequent category
    most_frequent_category = data['Electrical'].mode()[0]
    data['Electrical'].fillna(most_frequent_category, inplace=True)

    # Fill missing values in 'LotFrontage' with mean of neighborhood
    data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
    
    # Drop 'GarageYrBlt' column
    data = data.drop(columns=['GarageYrBlt'])
    
    # Fill missing values in 'MasVnrArea'
    data['MasVnrArea'].fillna(0.0, inplace=True)
    
    # Create new binary columns
    data['HasMasVnr'] = (data['MasVnrArea'] > 0).astype(int)
    data['Has2ndFlr'] = (data['2ndFlrSF'] > 0).astype(int)
    data['HasWoodDeck'] = (data['WoodDeckSF'] > 0).astype(int)
    data['HasEnclosedPorch'] = (data['EnclosedPorch'] > 0).astype(int)
    data['HasBsmtFinSF2'] = (data['BsmtFinSF2'] > 0).astype(int)

    # Combine porch features into one column
    data['TotPorch'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
    
    # Drop unnecessary columns after binary features are created
    data.drop(['BsmtFinSF2', 'BsmtHalfBath', 'ScreenPorch'], axis=1, inplace=True)
    data.drop(['PoolArea', 'MiscVal', '3SsnPorch', 'LowQualFinSF'], axis=1, inplace=True)
    
    # Remove outliers
    outlier_cols = ['LotArea', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
                    'WoodDeckSF', 'TotPorch']
    
    for col in outlier_cols:
        non_zero_values = data[data[col] > 0][col]
        mean = non_zero_values.mean()
        stddev = non_zero_values.std()
        threshold = mean + 3 * stddev
        data = data[data[col] <= threshold]
    
    # Drop columns with only 1 significant category
    columns_to_remove = ['Street', 'Utilities', 'Condition2', 'BsmtFinType2', 'RoofMatl', 'Heating', 
                         'Electrical', 'Functional', 'SaleType']
    data = data.drop(columns=columns_to_remove)

    # Binary encoding for binary features
    data['CentralAir'] = data['CentralAir'].map({'Y': 1, 'N': 0})
    
    # Map ordinal columns
    ord_c = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 
             'GarageQual', 'GarageCond']
    quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    
    for col in ord_c:
        data[col] = data[col].map(quality_mapping)
    
    ord_c2 = ['BsmtExposure', 'BsmtFinType1']
    quality_mapping2 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,
                        'ALQ': 5, 'GLQ': 6}
    
    for col in ord_c2:
        data[col] = data[col].map(quality_mapping2)
    
    # Combine rare categories
    def combine_rare_categories(df, column, threshold=20):
        category_counts = df[column].value_counts()
        rare_categories = category_counts[category_counts < threshold].index
        df[column] = df[column].apply(lambda x: 'Other' if x in rare_categories else x)
        return df
    
    columns_to_adjust = data.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_adjust:
        data = combine_rare_categories(data, col)
    
    # Drop additional columns
    data.drop(columns=['GarageCond', 'Fireplaces', 'GarageArea', 'BsmtFinSF1', 'TotRmsAbvGrd'], inplace=True)
    
    # Feature engineering
    data['TotalLivingSF'] = data['1stFlrSF'] + data['TotalBsmtSF']
    data.drop(columns=['1stFlrSF', 'TotalBsmtSF'], inplace=True)
    
    data['OverallQual_GrLivArea'] = data['OverallQual'] * data['GrLivArea']
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    
    from sklearn.preprocessing import LabelEncoder
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}

    # Apply Label Encoding to each categorical column
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le


    return data
