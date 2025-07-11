import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import json

np.random.seed(42)
tf.random.set_seed(42)


class HeMiRCAAnomalyScorer:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.autoencoder = None
        self.encoder = None

    def read_vector_file(self, file_path):
        """Read trace vector file and return timestamps and vectors"""
        print(f"Reading data from: {file_path}")
        vectors = []
        timestamps = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    if ':' not in line:
                        print(f"Warning: Line {line_num} doesn't contain ':' separator")
                        continue

                    timestamp, vec_str = line.split(":", 1)  # Split only on first ':'
                    vector = np.array([float(x) for x in vec_str.split(",")])
                    timestamps.append(int(timestamp))
                    vectors.append(vector)

                except ValueError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue

        if not vectors:
            raise ValueError(f"No valid vectors found in {file_path}")

        print(f"Loaded {len(vectors)} vectors with {len(vectors[0])} dimensions each")
        return np.array(timestamps), np.array(vectors)

    def build_autoencoder(self, input_dim):
        """Build autoencoder model similar to HeMiRCA approach"""
        print(f"Building autoencoder with input dimension: {input_dim}")

        # Encoder
        input_layer = Input(shape=(input_dim,))

        # Gradually reduce dimensions - adjust based on your service count
        encoded = Dense(min(128, input_dim), activation='relu')(input_layer)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(min(64, input_dim // 2), activation='relu')(encoded)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(min(32, input_dim // 4), activation='relu')(encoded)

        # Decoder - mirror the encoder
        decoded = Dense(min(64, input_dim // 2), activation='relu')(encoded)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(min(128, input_dim), activation='relu')(decoded)
        decoded = Dropout(0.1)(decoded)
        output_layer = Dense(input_dim, activation='linear')(decoded)  # Changed to linear for better reconstruction

        # Build models
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=encoded)

        # Compile with custom optimizer
        optimizer = Adam(learning_rate=0.001)
        self.autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        print("Autoencoder architecture:")
        self.autoencoder.summary()

        return self.autoencoder

    def train_autoencoder(self, X_train):
        """Train the autoencoder on normal data"""
        print("Training autoencoder on normal data...")

        # Callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Training
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=100,  # Increased epochs with early stopping
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_history.png'))
        plt.show()

    def calculate_anomaly_scores(self, X_test, timestamps_test):
        print("Calculating anomaly scores...")

        X_pred = self.autoencoder.predict(X_test, verbose=1)

        # Calculate reconstruction errors (MSE per sample)
        reconstruction_errors = np.mean(np.power(X_test - X_pred, 2), axis=1)

        # Also calculate MAE for comparison
        mae_errors = np.mean(np.abs(X_test - X_pred), axis=1)

        # Get encoded representations for additional analysis
        encoded_features = self.encoder.predict(X_test, verbose=1)

        # Create comprehensive results
        results = {
            'timestamps': timestamps_test,
            'mse_scores': reconstruction_errors,
            'mae_scores': mae_errors,
            'encoded_features': encoded_features
        }

        return results

    def save_results(self, results, output_file):
        print(f"Saving results to: {output_file}")

        # Create DataFrame with results
        df_out = pd.DataFrame({
            'timestamp': results['timestamps'],
            'mse_score': results['mse_scores'],
            'mae_score': results['mae_scores']
        })

        # Sort by timestamp
        df_out.sort_values(by='timestamp', inplace=True)

        # Save to CSV
        df_out.to_csv(output_file, index=False)

        # Also save summary statistics
        summary_file = output_file.replace('.csv', '_summary.json')
        summary = {
            'total_samples': len(results['timestamps']),
            'mse_statistics': {
                'mean': float(np.mean(results['mse_scores'])),
                'std': float(np.std(results['mse_scores'])),
                'min': float(np.min(results['mse_scores'])),
                'max': float(np.max(results['mse_scores'])),
                'median': float(np.median(results['mse_scores'])),
                'percentile_95': float(np.percentile(results['mse_scores'], 95)),
                'percentile_99': float(np.percentile(results['mse_scores'], 99))
            },
            'mae_statistics': {
                'mean': float(np.mean(results['mae_scores'])),
                'std': float(np.std(results['mae_scores'])),
                'min': float(np.min(results['mae_scores'])),
                'max': float(np.max(results['mae_scores'])),
                'median': float(np.median(results['mae_scores'])),
                'percentile_95': float(np.percentile(results['mae_scores'], 95)),
                'percentile_99': float(np.percentile(results['mae_scores'], 99))
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_file}")
        print(f"MSE Score Statistics - Mean: {summary['mse_statistics']['mean']:.6f}, "
              f"Std: {summary['mse_statistics']['std']:.6f}")

    def plot_anomaly_scores(self, results, output_dir):
        """Plot anomaly scores over time"""
        plt.figure(figsize=(15, 8))

        # Convert timestamps to readable format for plotting
        timestamps = results['timestamps']
        mse_scores = results['mse_scores']
        mae_scores = results['mae_scores']

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, mse_scores, 'b-', alpha=0.7, linewidth=1)
        plt.title('MSE Anomaly Scores Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('MSE Score')
        plt.grid(True, alpha=0.3)

        # Add threshold line (95th percentile)
        threshold_95 = np.percentile(mse_scores, 95)
        plt.axhline(y=threshold_95, color='r', linestyle='--', alpha=0.7,
                    label=f'95th percentile: {threshold_95:.6f}')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(timestamps, mae_scores, 'g-', alpha=0.7, linewidth=1)
        plt.title('MAE Anomaly Scores Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('MAE Score')
        plt.grid(True, alpha=0.3)

        # Add threshold line (95th percentile)
        threshold_95_mae = np.percentile(mae_scores, 95)
        plt.axhline(y=threshold_95_mae, color='r', linestyle='--', alpha=0.7,
                    label=f'95th percentile: {threshold_95_mae:.6f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_scores_timeline.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def run_anomaly_detection(self):
        """Main function to run the complete anomaly detection pipeline"""
        print("Starting HeMiRCA-style anomaly detection...")

        # Load data
        print("\n1. Loading data...")
        _, X_train = self.read_vector_file(self.config['normal_file'])
        timestamps_test, X_test = self.read_vector_file(self.config['abnormal_file'])

        print(f"Normal data shape: {X_train.shape}")
        print(f"Abnormal data shape: {X_test.shape}")

        # Normalize data
        print("\n2. Normalizing data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build and train autoencoder
        print("\n3. Building and training autoencoder...")
        input_dim = X_train.shape[1]
        self.build_autoencoder(input_dim)
        self.train_autoencoder(X_train_scaled)

        # Calculate anomaly scores
        print("\n4. Calculating anomaly scores...")
        results = self.calculate_anomaly_scores(X_test_scaled, timestamps_test)

        # Save results
        print("\n5. Saving results...")
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.save_results(results, self.config['score_output_file'])

        # Plot results
        print("\n6. Creating visualizations...")
        self.plot_anomaly_scores(results, self.config['output_dir'])

        print("\nAnomaly detection completed successfully!")
        return results


def main():
    """Main execution function"""
    # Configuration - adjust these paths to match your setup
    config = {
        'normal_file': r'D:\HeMiRCA\vector\trace_vector_normal.txt',
        'abnormal_file': r'D:\HeMiRCA\vector\trace_vector_abnormal.txt',
        'score_output_file': r'D:\HeMiRCA\vector\abnormal_trace_scores.csv',
        'output_dir': r'D:\HeMiRCA\results'
    }

    # Initialize and run anomaly detection
    scorer = HeMiRCAAnomalyScorer(config)
    results = scorer.run_anomaly_detection()

    # Print final summary
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total abnormal samples analyzed: {len(results['timestamps'])}")
    print(f"MSE scores - Mean: {np.mean(results['mse_scores']):.6f}, "
          f"Max: {np.max(results['mse_scores']):.6f}")
    print(f"Results saved to: {config['score_output_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

    output_txt = r"D:\HeMiRCA\vector\abnormal_trace_scores-mse.txt"
    # Read the CSV
    df = pd.read_csv(r"D:\HeMiRCA\vector\abnormal_trace_scores.csv")
    # Choose one of the error columns (e.g., mse_score)
    df_output = df[['timestamp', 'mse_score']].copy()
    df_output.rename(columns={'mse_score': 'score'}, inplace=True)
    df_output.to_csv(output_txt, index=False)
    print("Conversion completed. Saved to:", output_txt)