"""
Real-time CTR/CVR Prediction Model with LightGBM + Deep Learning Hybrid Architecture

This module implements the core prediction models for click-through rate (CTR) and 
conversion rate (CVR) prediction in a real-time bidding environment.

Main components:
1. Feature preprocessing pipeline
2. LightGBM model for feature interactions
3. Deep neural network for final prediction
4. Model training and inference logic
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

# Data processing and ML libraries
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# For feature engineering
from scipy.sparse import csr_matrix
import category_encoders as ce

# ===== Configuration =====
class ModelConfig:
    """Configuration for the hybrid model"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with default values or from config file"""
        # Default configuration values
        self.seed = 42
        self.train_data_path = "data/criteo/train.csv"
        self.test_data_path = "data/criteo/test.csv"
        self.model_save_path = "models/saved/"
        
        # Feature configuration
        self.num_features = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5", 
                            "feat_6", "feat_7", "feat_8", "feat_9", "feat_10", 
                            "feat_11", "feat_12", "feat_13"]
        self.cat_features = [f"C{i}" for i in range(1, 27)]  # C1-C26
        
        # LightGBM parameters
        self.lgb_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'num_leaves': 255,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 8
        }
        
        # Deep model parameters
        self.dnn_hidden_units = [512, 256, 128, 64]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 1024
        self.epochs = 5
        
        # Runtime parameters
        self.use_gpu = True
        self.gpu_device = 0
        
        # Override defaults with config file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
            
    def _load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def save(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {key: value for key, value in self.__dict__.items() 
                      if not key.startswith('_') and not callable(value)}
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# ===== Feature Engineering =====
class FeatureProcessor:
    """Feature preprocessing for CTR/CVR prediction"""
    
    def __init__(self, config: ModelConfig):
        """Initialize feature processor with configuration"""
        self.config = config
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        
        # Feature transformation components
        self.num_scaler = StandardScaler()
        self.cat_encoders = {}
        self.feature_names = []
        self.cat_dimensions = {}  # Store dimension sizes for each categorical feature
        
    def fit(self, df: pd.DataFrame):
        """Fit feature processors on training data"""
        print("Fitting feature processors...")
        
        # Fit numerical feature scaler
        if self.num_features:
            # Fill missing values with mean
            for feat in self.num_features:
                if df[feat].isnull().sum() > 0:
                    df[feat] = df[feat].fillna(df[feat].mean())
            
            self.num_scaler.fit(df[self.num_features])
        
        # Fit categorical encoders
        for cat_feat in self.cat_features:
            # Fill missing values with a special value
            df[cat_feat] = df[cat_feat].fillna("MISSING")
            
            # Use label encoding for categorical features
            encoder = LabelEncoder()
            encoder.fit(df[cat_feat])
            self.cat_encoders[cat_feat] = encoder
            self.cat_dimensions[cat_feat] = len(encoder.classes_)
            
        # Store feature names for later use
        self.feature_names = self.num_features + self.cat_features
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Transform features for model input
        
        Returns:
            - Numerical features array for LightGBM
            - Dictionary of categorical features for deep model
        """
        # Handle numerical features
        num_features = None
        if self.num_features:
            # Fill missing values with mean
            for feat in self.num_features:
                if df[feat].isnull().sum() > 0:
                    df[feat] = df[feat].fillna(df[feat].mean())
            
            num_features = self.num_scaler.transform(df[self.num_features])
        
        # Handle categorical features
        cat_features = {}
        for cat_feat in self.cat_features:
            # Fill missing values
            df[cat_feat] = df[cat_feat].fillna("MISSING")
            
            # Encode categorical values
            encoder = self.cat_encoders[cat_feat]
            # Handle unseen categories
            valid_cats = np.isin(df[cat_feat].values, encoder.classes_)
            cat_values = df[cat_feat].values.copy()
            cat_values[~valid_cats] = encoder.classes_[0]  # Default to first class for unseen values
            
            cat_features[cat_feat] = encoder.transform(cat_values)
        
        # Combine all features for GBDT
        all_features = num_features if num_features is not None else np.zeros((len(df), 0))
        
        return all_features, cat_features
    
    def save(self, path: str):
        """Save feature processor to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'num_scaler': self.num_scaler,
                'cat_encoders': self.cat_encoders,
                'feature_names': self.feature_names,
                'cat_dimensions': self.cat_dimensions
            }, f)
        
    def load(self, path: str):
        """Load feature processor from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.num_scaler = data['num_scaler']
            self.cat_encoders = data['cat_encoders']
            self.feature_names = data['feature_names']
            self.cat_dimensions = data['cat_dimensions']
        return self


# ===== GBDT Model Component =====
class GBDTModel:
    """LightGBM model for feature interactions and initial prediction"""
    
    def __init__(self, config: ModelConfig):
        """Initialize GBDT model with configuration"""
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train LightGBM model"""
        print("Training LightGBM model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            self.model = lgb.train(
                self.config.lgb_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=500,
                early_stopping_rounds=50,
                verbose_eval=50
            )
        else:
            self.model = lgb.train(
                self.config.lgb_params,
                train_data,
                num_boost_round=500
            )
            
        # Store feature importance
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from GBDT model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def get_leaf_output(self, X: np.ndarray) -> np.ndarray:
        """Extract leaf indices for feature transformation"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X, pred_leaf=True)
    
    def save(self, path: str):
        """Save GBDT model to disk"""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        
        # Save feature importance
        importance_path = os.path.join(os.path.dirname(path), 
                                      f"{os.path.basename(path)}.importance.json")
        with open(importance_path, 'w') as f:
            json.dump({
                'feature_importance': self.feature_importance.tolist()
            }, f)
        
    def load(self, path: str):
        """Load GBDT model from disk"""
        self.model = lgb.Booster(model_file=path)
        
        # Load feature importance if available
        importance_path = os.path.join(os.path.dirname(path), 
                                      f"{os.path.basename(path)}.importance.json")
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                data = json.load(f)
                self.feature_importance = np.array(data['feature_importance'])
                
        return self


# ===== Deep Learning Model Component =====
class DeepModel:
    """Deep neural network for final CTR/CVR prediction"""
    
    def __init__(self, config: ModelConfig, cat_dimensions: Dict[str, int]):
        """Initialize deep model with configuration
        
        Args:
            config: Model configuration
            cat_dimensions: Dictionary mapping categorical feature names to their cardinality
        """
        self.config = config
        self.cat_dimensions = cat_dimensions
        self.cat_feature_names = list(cat_dimensions.keys())
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build deep neural network architecture"""
        # Set up inputs
        num_input = layers.Input(shape=(len(self.config.num_features),), name='num_features')
        gbdt_input = layers.Input(shape=(1,), name='gbdt_pred')
        leaf_inputs = layers.Input(shape=(1,), name='gbdt_leaves')
        
        # Dictionary of categorical inputs
        cat_inputs = {}
        cat_embeddings = []
        
        # Create embeddings for each categorical feature
        for feat_name, dimension in self.cat_dimensions.items():
            # Add input layer for this categorical feature
            cat_inputs[feat_name] = layers.Input(shape=(1,), name=feat_name)
            
            # Create embedding for this feature
            # Embedding dimension is 6 * (dimension)^0.25 capped at 100
            embed_dim = min(100, int(6 * (dimension ** 0.25)))
            embedding = layers.Embedding(
                input_dim=dimension,
                output_dim=embed_dim,
                embeddings_regularizer=regularizers.l2(1e-5),
                name=f"{feat_name}_embedding"
            )(cat_inputs[feat_name])
            
            # Flatten embedding
            embedding = layers.Flatten()(embedding)
            cat_embeddings.append(embedding)
        
        # Combine all features
        if cat_embeddings:
            cat_concat = layers.concatenate(cat_embeddings, name='cat_concatenate')
        else:
            cat_concat = layers.Lambda(lambda x: tf.zeros((tf.shape(num_input)[0], 1)))(num_input)
            
        # Combine with numerical features and GBDT outputs
        combined = layers.concatenate([num_input, cat_concat, gbdt_input, leaf_inputs], 
                                     name='combined_features')
        
        # Deep network
        x = combined
        for i, units in enumerate(self.config.dnn_hidden_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        
        # Create model with all inputs
        inputs = [num_input, gbdt_input, leaf_inputs] + list(cat_inputs.values())
        model = models.Model(inputs=inputs, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['AUC', 'binary_crossentropy']
        )
        
        return model
    
    def train(self, X_train: Dict, y_train: np.ndarray, 
              X_val: Dict = None, y_val: np.ndarray = None):
        """Train deep model
        
        Args:
            X_train: Dictionary of training features
            y_train: Training labels
            X_val: Dictionary of validation features
            y_val: Validation labels
        """
        print("Training deep model...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=2
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            x=X_train,
            y=y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: Dict) -> np.ndarray:
        """Get predictions from deep model"""
        return self.model.predict(X, batch_size=self.config.batch_size * 4)
    
    def save(self, path: str):
        """Save deep model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save categorical dimensions
        meta_path = os.path.join(os.path.dirname(path), 
                               f"{os.path.basename(path)}.meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                'cat_dimensions': self.cat_dimensions,
                'cat_feature_names': self.cat_feature_names
            }, f)
        
    def load(self, path: str):
        """Load deep model from disk"""
        self.model = tf.keras.models.load_model(path)
        
        # Load metadata
        meta_path = os.path.join(os.path.dirname(path), 
                               f"{os.path.basename(path)}.meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data = json.load(f)
                self.cat_dimensions = data['cat_dimensions']
                self.cat_feature_names = data['cat_feature_names']
        
        return self


# ===== Hybrid CTR/CVR Model =====
class HybridCTRCVRModel:
    """Hybrid model combining GBDT and Deep Learning for CTR/CVR prediction"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize hybrid model with configuration"""
        self.config = ModelConfig(config_path)
        self.feature_processor = FeatureProcessor(self.config)
        self.gbdt_model = GBDTModel(self.config)
        self.deep_model = None  # Will be initialized after feature processing
        
    def _prepare_deep_model_input(self, X_num: np.ndarray, X_cat: Dict[str, np.ndarray], 
                                 gbdt_pred: np.ndarray, gbdt_leaves: np.ndarray) -> Dict:
        """Prepare input for deep model"""
        inputs = {
            'num_features': X_num,
            'gbdt_pred': gbdt_pred.reshape(-1, 1),
            'gbdt_leaves': gbdt_leaves[:, 0].reshape(-1, 1)  # Only use first tree for simplicity
        }
        
        # Add categorical features
        for feat_name in self.feature_processor.cat_features:
            inputs[feat_name] = X_cat[feat_name].reshape(-1, 1)
            
        return inputs
    
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the complete hybrid model
        
        Args:
            train_df: Training dataframe with features and label columns
            val_df: Optional validation dataframe
        """
        print(f"Training hybrid CTR/CVR model with {len(train_df)} samples...")
        
        # Extract labels
        y_train = train_df['label'].values
        y_val = val_df['label'].values if val_df is not None else None
        
        # Process features
        self.feature_processor.fit(train_df)
        X_train_num, X_train_cat = self.feature_processor.transform(train_df)
        
        if val_df is not None:
            X_val_num, X_val_cat = self.feature_processor.transform(val_df)
        else:
            X_val_num, X_val_cat = None, None
        
        # Train GBDT model
        self.gbdt_model.train(X_train_num, y_train, X_val_num, y_val)
        
        # Get GBDT outputs for deep model input
        gbdt_train_pred = self.gbdt_model.predict(X_train_num)
        gbdt_train_leaves = self.gbdt_model.get_leaf_output(X_train_num)
        
        if val_df is not None:
            gbdt_val_pred = self.gbdt_model.predict(X_val_num)
            gbdt_val_leaves = self.gbdt_model.get_leaf_output(X_val_num)
        
        # Initialize and train deep model
        self.deep_model = DeepModel(self.config, self.feature_processor.cat_dimensions)
        
        # Prepare inputs for deep model
        deep_train_input = self._prepare_deep_model_input(
            X_train_num, X_train_cat, gbdt_train_pred, gbdt_train_leaves)
        
        if val_df is not None:
            deep_val_input = self._prepare_deep_model_input(
                X_val_num, X_val_cat, gbdt_val_pred, gbdt_val_leaves)
            
            self.deep_model.train(deep_train_input, y_train, deep_val_input, y_val)
        else:
            self.deep_model.train(deep_train_input, y_train)
            
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate CTR/CVR predictions for new data
        
        Args:
            df: Dataframe with feature columns
            
        Returns:
            Array of predicted probabilities
        """
        # Process features
        X_num, X_cat = self.feature_processor.transform(df)
        
        # Get GBDT outputs
        gbdt_pred = self.gbdt_model.predict(X_num)
        gbdt_leaves = self.gbdt_model.get_leaf_output(X_num)
        
        # Prepare deep model input
        deep_input = self._prepare_deep_model_input(X_num, X_cat, gbdt_pred, gbdt_leaves)
        
        # Get final predictions
        predictions = self.deep_model.predict(deep_input)
        
        return predictions.flatten()
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            df: Dataframe with feature columns and label column
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract labels
        y_true = df['label'].values
        
        # Get predictions
        y_pred = self.predict(df)
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        
        # Calculate time per prediction
        start_time = time.time()
        _ = self.predict(df.head(1000))
        end_time = time.time()
        time_per_pred = (end_time - start_time) / 1000 * 1000  # ms
        
        return {
            'auc': auc,
            'logloss': logloss,
            'time_per_pred_ms': time_per_pred
        }
    
    def save(self, directory: str):
        """Save complete model to directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(directory, 'config.json'))
        
        # Save feature processor
        self.feature_processor.save(os.path.join(directory, 'feature_processor.pkl'))
        
        # Save GBDT model
        self.gbdt_model.save(os.path.join(directory, 'gbdt_model.txt'))
        
        # Save deep model
        self.deep_model.save(os.path.join(directory, 'deep_model'))
        
        print(f"Model saved to {directory}")
        
    @staticmethod
    def load(directory: str) -> 'HybridCTRCVRModel':
        """Load complete model from directory"""
        # Load config
        config_path = os.path.join(directory, 'config.json')
        model = HybridCTRCVRModel(config_path)
        
        # Load feature processor
        model.feature_processor.load(os.path.join(directory, 'feature_processor.pkl'))
        
        # Load GBDT model
        model.gbdt_model.load(os.path.join(directory, 'gbdt_model.txt'))
        
        # Initialize and load deep model
        model.deep_model = DeepModel(model.config, model.feature_processor.cat_dimensions)
        model.deep_model.load(os.path.join(directory, 'deep_model'))
        
        print(f"Model loaded from {directory}")
        return model


# ===== Simple Training Script =====
def main():
    """Simple script to demonstrate model training and evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CTR/CVR prediction model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='models/hybrid_model', 
                       help='Output directory for model')
    
    args = parser.parse_args()
    
    # Load and prepare data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = HybridCTRCVRModel(args.config)
    model.fit(train_df, val_df)
    
    # Evaluate model
    metrics = model.evaluate(val_df)
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    
    # Save model
    model.save(args.output)


if __name__ == "__main__":
    main()
