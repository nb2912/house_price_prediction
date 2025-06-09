import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """
        Load and prepare the dataset
        """
        # Load the dataset
        self.data = pd.read_csv(file_path)
        print("Dataset loaded with shape:", self.data.shape)
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the data including handling missing values and feature engineering
        """
        print("Data preprocessing started...")
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Basic preprocessing steps
        
        # 1. Handle missing values
        self.data['price'] = pd.to_numeric(self.data['price'], errors='coerce')
        self.data['area'] = pd.to_numeric(self.data['area'], errors='coerce')
        self.data['price_per_sqft'] = pd.to_numeric(self.data['price_per_sqft'], errors='coerce')
        
        # 2. Extract numeric values from bedRoom and bathroom
        self.data['bedRoom'] = pd.to_numeric(self.data['bedRoom'], errors='coerce')
        self.data['bathroom'] = pd.to_numeric(self.data['bathroom'], errors='coerce')
        
        # 3. Create binary features from amenities
        self.data['has_gym'] = self.data['features'].str.contains('Fitness Centre / GYM', case=False, na=False).astype(int)
        self.data['has_pool'] = self.data['features'].str.contains('Swimming Pool', case=False, na=False).astype(int)
        self.data['has_security'] = self.data['features'].str.contains('Security Personnel', case=False, na=False).astype(int)
        self.data['has_park'] = self.data['features'].str.contains('Park', case=False, na=False).astype(int)
        
        # 4. Floor number is already numeric
        self.data['floor_num'] = self.data['floorNum']
        
        # 5. Create age category
        age_mapping = {
            '0 to 1 Year Old': 0.5,
            'Within 6 months': 0.25,
            '1 to 5 Year Old': 3,
            '5 to 10 Year Old': 7.5,
            '10+ Year Old': 12
        }
        self.data['property_age'] = self.data['agePossession'].map(age_mapping)
        
        # 6. Extract location information from address
        self.data['state'] = self.data['address'].fillna('Unknown').str.extract(r'([^,]+)$').iloc[:, 0].str.strip()
        self.data['city'] = self.data['address'].fillna('Unknown').str.extract(r'([^,]+),').iloc[:, 0].str.strip()
        
        # 7. Extract or create area/locality from address
        self.data['locality'] = self.data['address'].fillna('Unknown').str.extract(r'([^,]+),.*,').iloc[:, 0].str.strip()
        
        # 8. Extract pincode if available, otherwise mark as unknown
        self.data['pincode'] = self.data['address'].str.extract(r'(\d{6})').iloc[:, 0]
        
        # 9. Encode categorical variables
        categorical_columns = ['property_type', 'facing', 'state', 'city', 'locality']
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col + '_encoded'] = le.fit_transform(self.data[col].fillna('Unknown'))
                self.label_encoders[col] = le
        
        print("\nPreprocessing completed.")
        
    def perform_eda(self):
        """
        Perform Exploratory Data Analysis
        """
        print("Performing EDA...")
        
        # 1. Price Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['price'].dropna(), bins=50)
        plt.title('Price Distribution')
        plt.xlabel('Price (Crores)')
        plt.ylabel('Count')
        plt.savefig('plots/price_distribution.png')
        plt.close()
        
        # 2. Price vs Area
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['area'], self.data['price'], alpha=0.5)
        plt.title('Price vs Area')
        plt.xlabel('Area (sq ft)')
        plt.ylabel('Price (Crores)')
        plt.savefig('plots/price_vs_area.png')
        plt.close()
        
        # 3. Average Price by Bedroom Count
        plt.figure(figsize=(10, 6))
        avg_price_by_bedroom = self.data.groupby('bedRoom')['price'].mean()
        avg_price_by_bedroom[avg_price_by_bedroom.index <= 10].plot(kind='bar')
        plt.title('Average Price by Number of Bedrooms')
        plt.xlabel('Number of Bedrooms')
        plt.ylabel('Average Price (Crores)')
        plt.savefig('plots/avg_price_by_bedrooms.png')
        plt.close()
        
        # 4. Average Price by City
        plt.figure(figsize=(12, 6))
        city_prices = self.data.groupby('city')['price'].mean().sort_values(ascending=False).head(10)
        city_prices.plot(kind='bar')
        plt.title('Average Price by Top 10 Cities')
        plt.xlabel('City')
        plt.ylabel('Average Price (Crores)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/avg_price_by_city.png')
        plt.close()
        
        # 5. Correlation Matrix
        numeric_columns = ['price', 'area', 'bedRoom', 'bathroom', 'price_per_sqft', 
                         'has_gym', 'has_pool', 'has_security', 'has_park', 'floor_num']
        correlation_matrix = self.data[numeric_columns].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.savefig('plots/correlation_matrix.png')
        plt.close()
        
    def prepare_features(self):
        """
        Prepare features for modeling
        """
        # Select features for modeling
        feature_columns = [
            'area', 'bedRoom', 'bathroom', 'price_per_sqft', 'has_gym', 
            'has_pool', 'has_security', 'has_park', 'floor_num', 'property_age',
            'property_type_encoded', 'facing_encoded', 'state_encoded', 
            'city_encoded', 'locality_encoded'
        ]
        
        # Remove rows with missing values
        self.data_clean = self.data.dropna(subset=['price'] + feature_columns)
        
        # Remove outliers
        def remove_outliers(df, column, n_std):
            mean = df[column].mean()
            std = df[column].std()
            return df[(df[column] >= mean - n_std * std) & (df[column] <= mean + n_std * std)]
        
        # Remove extreme outliers for key numeric columns
        self.data_clean = remove_outliers(self.data_clean, 'price', 3)
        self.data_clean = remove_outliers(self.data_clean, 'area', 3)
        self.data_clean = remove_outliers(self.data_clean, 'price_per_sqft', 3)
        
        # Prepare features and target
        self.X = self.data_clean[feature_columns]
        self.y = self.data_clean['price']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_model(self):
        """
        Train the model
        """
        # Initialize and train the model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        train_predictions = self.model.predict(self.X_train_scaled)
        test_predictions = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))
        train_mae = mean_absolute_error(self.y_train, train_predictions)
        test_mae = mean_absolute_error(self.y_test, test_predictions)
        train_r2 = r2_score(self.y_train, train_predictions)
        test_r2 = r2_score(self.y_test, test_predictions)
        
        print(f"\nModel Performance:")
        print(f"Train RMSE: {train_rmse:.2f} Crores")
        print(f"Test RMSE: {test_rmse:.2f} Crores")
        print(f"Train MAE: {train_mae:.2f} Crores")
        print(f"Test MAE: {test_mae:.2f} Crores")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
        
    def plot_results(self):
        """
        Plot actual vs predicted values
        """
        test_predictions = self.model.predict(self.X_test_scaled)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, test_predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Prices (Crores)')
        plt.ylabel('Predicted Prices (Crores)')
        plt.title('Actual vs Predicted House Prices')
        plt.tight_layout()
        plt.savefig('plots/actual_vs_predicted.png')
        plt.close()
        
    def predict_price(self, features_dict):
        """
        Predict price for new data
        """
        # Create a DataFrame with the input features
        input_df = pd.DataFrame([features_dict])
        
        # Ensure all required columns are present
        for col in self.X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Scale the features
        input_scaled = self.scaler.transform(input_df[self.X.columns])
        
        # Make prediction
        predicted_price = self.model.predict(input_scaled)[0]
        
        return predicted_price

    def get_unique_locations(self):
        """
        Get unique values for location-based features
        """
        return {
            'states': sorted([str(x) for x in self.data['state'].unique() if pd.notna(x)]),
            'cities': sorted([str(x) for x in self.data['city'].unique() if pd.notna(x)]),
            'localities': sorted([str(x) for x in self.data['locality'].unique() if pd.notna(x)])
        }

def main():
    # Initialize the predictor
    predictor = HousePricePredictor()
    
    # Load data
    try:
        data = predictor.load_data('data/house_cleaned.csv')
    except FileNotFoundError:
        print("Please ensure the dataset is in the 'data' directory with name 'house_cleaned.csv'")
        return
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Perform EDA
    predictor.perform_eda()
    
    # Prepare features
    predictor.prepare_features()
    
    # Train model
    predictor.train_model()
    
    # Plot results
    predictor.plot_results()

if __name__ == "__main__":
    main() 