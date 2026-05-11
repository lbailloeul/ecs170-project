#!/usr/bin/env python3
"""
CSV-based CNN Trainer for SPS Histogram Classification

This script loads histograms from ROOT files using CSV ratings data
and trains a CNN model for good/bad histogram classification.

Author: Leo Bailloeul, Claude Code Assistant
Date: 2025-09-04
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TensorFlow setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


class CSVHistogramLoader:
    """Load histograms from ROOT file using CSV ratings data"""

    def __init__(self, root_file_path: str, csv_ratings_path: str):
        self.root_file_path = Path(root_file_path)
        self.csv_ratings_path = Path(csv_ratings_path)

        if not self.root_file_path.exists():
            raise FileNotFoundError(f"ROOT file not found: {self.root_file_path}")
        if not self.csv_ratings_path.exists():
            raise FileNotFoundError(
                f"CSV ratings file not found: {self.csv_ratings_path}"
            )

        # Import uproot for ROOT file reading
        try:
            import uproot

            self.uproot = uproot
        except ImportError:
            raise ImportError("uproot is required. Install with: pip install uproot")

    def load_ratings_data(self) -> pd.DataFrame:
        """Load and process the ratings CSV file"""
        logger.info(f"Loading ratings from {self.csv_ratings_path}")

        df = pd.read_csv(self.csv_ratings_path)
        logger.info(f"Loaded {len(df)} total ratings")

        # Filter out unrated histograms (empty rating field)
        df_rated = df[df["rating"].notna() & (df["rating"] != "")]
        logger.info(f"Found {len(df_rated)} rated histograms")

        # Convert ratings to binary labels
        df_rated = df_rated.copy()

        # Convert ratings to binary: 1-2 = Bad (0), 3-5 = Good (1), BAD_AUTO = Bad (0), GOOD = Good (1)
        def rating_to_binary(rating):
            if rating in ["BAD_AUTO", "BAD"]:
                return 0
            elif rating in ["1", "2", 1, 2]:
                return 0  # Bad
            elif rating in ["3", "4", "5", 3, 4, 5, "GOOD"]:
                return 1  # Good
            else:
                logger.warning(f"Unknown rating: {rating}")
                return None

        df_rated["binary_label"] = df_rated["rating"].apply(rating_to_binary)
        df_rated = df_rated.dropna(subset=["binary_label"])

        logger.info(f"Rating distribution:")
        logger.info(
            f"  Good (3-5): {np.sum(df_rated['binary_label'] == 1)} ({np.sum(df_rated['binary_label'] == 1)/len(df_rated)*100:.1f}%)"
        )
        logger.info(
            f"  Bad (1-2, BAD, BAD_AUTO): {np.sum(df_rated['binary_label'] == 0)} ({np.sum(df_rated['binary_label'] == 0)/len(df_rated)*100:.1f}%)"
        )

        return df_rated

    def load_histograms_and_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load histograms from ROOT file based on CSV ratings with caching"""

        # Check for cached data (z-score extended version)
        csv_bytes = self.csv_ratings_path.read_bytes()
        csv_hash = hashlib.md5(csv_bytes).hexdigest()[:10]
        cache_file = Path(f"histogram_data_cache_zscore_extended_{csv_hash}.npz")

        if cache_file.exists():
            logger.info("Loading from cache...")
            cached = np.load(cache_file, allow_pickle=True)
            return cached["X"], cached["y"], cached["names"]

        # Load ratings data
        ratings_df = self.load_ratings_data()
        logger.info(f"Loading {len(ratings_df)} histograms from {self.root_file_path}")

        # Batch load all histograms at once
        with self.uproot.open(self.root_file_path) as f:
            logger.info(f"ROOT file contains {len(f.keys())} objects")

            # Get all histogram names we need
            needed_hists = set(ratings_df["histogram_name"].values)
            available_keys = {key.split(";")[0]: key for key in f.keys()}

            logger.info(f"Batch loading {len(needed_hists)} histograms...")

            # Load all needed histograms in one operation
            hist_data = {}
            for hist_name in needed_hists:
                if hist_name in available_keys:
                    try:
                        hist = f[available_keys[hist_name]]
                        values, edges = hist.to_numpy()
                        hist_data[hist_name] = values.astype(np.float32)
                    except Exception as e:
                        logger.warning(f"Failed to load {hist_name}: {e}")
                        continue

        logger.info(f"Successfully loaded {len(hist_data)} histograms from ROOT file")

        # Process histograms according to ratings
        histograms = []
        labels = []
        names = []
        target_length = 300

        for _, row in ratings_df.iterrows():
            hist_name = row["histogram_name"]
            label = int(row["binary_label"])

            if hist_name in hist_data:
                values = hist_data[hist_name]

                # Pad or truncate to standard length
                if len(values) != target_length:
                    if len(values) > target_length:
                        values = values[:target_length]
                    else:
                        padded = np.zeros(target_length, dtype=np.float32)
                        padded[: len(values)] = values
                        values = padded

                # Z-score normalization per histogram
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val > 0:
                    values = (values - mean_val) / std_val
                else:
                    # Handle case where std is 0 (constant histogram)
                    values = values - mean_val

                histograms.append(values)
                labels.append(label)
                names.append(hist_name)

        X = np.array(histograms, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Cache the results for next time
        logger.info("Caching processed data...")
        np.savez_compressed(cache_file, X=X, y=y, names=names)

        logger.info(f"Successfully loaded dataset:")
        logger.info(f"  Total histograms: {len(X)}")
        logger.info(
            f"  Good histograms: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)"
        )
        logger.info(
            f"  Bad histograms: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)"
        )
        logger.info(f"  Histogram shape: {X.shape}")

        if len(X) == 0:
            raise ValueError("No histograms were successfully loaded!")

        return X, y, names


class CSVCNNTrainer:
    """CNN trainer that works with CSV ratings and ROOT histograms"""

    def __init__(self, output_dir: str = "csv_training_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.history = None

        logger.info(f"CSV CNN Trainer initialized. Output: {self.output_dir}")

    def build_model(self, input_shape: int) -> keras.Model:
        """Build CNN model for histogram classification"""

        model = keras.Sequential(
            [
                # Input layer
                layers.Input(shape=(input_shape,)),
                # Reshape for 1D CNN
                layers.Reshape((input_shape, 1)),
                # 1D Convolutional layers
                layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2),
                layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2),
                layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2),
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                # Output layer
                layers.Dense(2, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info("Model built successfully")
        model.summary(print_fn=logger.info)

        return model

    def setup_callbacks(self, patience: int = 15):
        """Setup training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=8, min_lr=1e-7, verbose=1
            ),
        ]

        return callbacks

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        test_size: float = 0.2,
    ) -> Dict:
        """Train the model on histogram data"""

        logger.info("Preparing data for training...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        logger.info(
            f"Train labels - Good: {np.sum(y_train == 1)}, Bad: {np.sum(y_train == 0)}"
        )
        logger.info(
            f"Val labels - Good: {np.sum(y_val == 1)}, Bad: {np.sum(y_val == 0)}"
        )

        # Build model
        self.model = self.build_model(input_shape=X.shape[1])

        # Setup callbacks
        callbacks = self.setup_callbacks(patience=15)

        # Calculate class weights automatically
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        logger.info(f"Calculated class weights: {class_weight_dict}")

        # Train model
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = datetime.now()

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")

        # Evaluate model
        val_accuracy = max(self.history.history["val_accuracy"])
        logger.info(f"Best validation accuracy: {val_accuracy:.4f}")

        # Save training artifacts and metadata
        self.save_training_artifacts(X_train, X_val, y_train, y_val)

        return {
            "best_val_accuracy": val_accuracy,
            "training_time": str(training_time),
            "history": self.history.history,
        }

    def load_model(self, model_path: str) -> keras.Model:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        logger.info(f"Loading model from {model_path}")

        # Keras ≥ 2.12 removed renorm_* fields from BatchNormalization.
        # This shim lets .h5 files saved with older Keras load cleanly.
        from tensorflow.keras.layers import BatchNormalization as _BN
        class _BNCompat(_BN):
            def __init__(self, **kwargs):
                for k in ("renorm", "renorm_clipping", "renorm_momentum"):
                    kwargs.pop(k, None)
                super().__init__(**kwargs)

        self.model = keras.models.load_model(
            model_path,
            custom_objects={"BatchNormalization": _BNCompat},
        )
        self.model.summary(print_fn=logger.info)
        return self.model

    def fine_tune(
            self,
            X: np.ndarray,
            y: np.ndarray,
            model_path: str,
            epochs: int = 30,
            batch_size: int = 64,
            test_size: float = 0.2,
            learning_rate: float = 1e-4,
    ) -> Dict:
        logger.info("Preparing data for fine-tuning...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Load existing model
        self.load_model(model_path)

        # Re-compile with a smaller LR for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = self.setup_callbacks(patience=10)

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        logger.info(f"Calculated class weights: {class_weight_dict}")

        logger.info(f"Starting fine-tuning for {epochs} epochs at lr={learning_rate}...")
        start_time = datetime.now()

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = datetime.now() - start_time
        val_accuracy = max(self.history.history["val_accuracy"])

        logger.info(f"Fine-tuning completed in {training_time}")
        logger.info(f"Best validation accuracy: {val_accuracy:.4f}")

        self.save_training_artifacts(X_train, X_val, y_train, y_val)

        return {
            "best_val_accuracy": val_accuracy,
            "training_time": str(training_time),
            "history": self.history.history,
        }

    def save_training_artifacts(self, X_train, X_val, y_train, y_val):
        """Save training history and evaluation results"""

        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(self.output_dir / "training_history.csv", index=False)

        # Create training plots
        self.plot_training_history()

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Classification report
        report = classification_report(
            y_val, y_pred_classes, target_names=["bad", "good"], output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)

        # Save evaluation results
        eval_results = {
            "validation_accuracy": float(report["accuracy"]),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "dataset_info": {
                "total_samples": len(X_train) + len(X_val),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "train_good": int(np.sum(y_train == 1)),
                "train_bad": int(np.sum(y_train == 0)),
                "val_good": int(np.sum(y_val == 1)),
                "val_bad": int(np.sum(y_val == 0)),
            },
        }

        with open(self.output_dir / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2, default=str)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm)

        logger.info(f"Training artifacts saved to {self.output_dir}")
        logger.info(f"Final validation accuracy: {report['accuracy']:.4f}")
        logger.info(
            f"Precision - Good: {report['good']['precision']:.3f}, Bad: {report['bad']['precision']:.3f}"
        )
        logger.info(
            f"Recall - Good: {report['good']['recall']:.3f}, Bad: {report['bad']['recall']:.3f}"
        )

    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy plot
        ax1.plot(self.history.history["accuracy"], label="Training Accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(self.history.history["loss"], label="Training Loss")
        ax2.plot(self.history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        # Learning rate plot (if available)
        if "lr" in self.history.history:
            ax3.plot(self.history.history["lr"], label="Learning Rate")
            ax3.set_title("Learning Rate")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.legend()
            ax3.grid(True)
        else:
            ax3.text(
                0.5,
                0.5,
                "Learning Rate\nNot Available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )

        # Loss zoomed plot
        ax4.plot(self.history.history["loss"][2:], label="Training Loss (from epoch 3)")
        ax4.plot(
            self.history.history["val_loss"][2:], label="Validation Loss (from epoch 3)"
        )
        ax4.set_title("Model Loss (Zoomed)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_history.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Bad", "Good"],
            yticklabels=["Bad", "Good"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(
            self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train CNN on CSV-rated histogram data"
    )
    parser.add_argument(
        "--root_file",
        default="data/extracted_sps_histograms_smart.root",
        help="Path to ROOT file with histograms",
    )
    parser.add_argument(
        "--csv_ratings",
        default="data/histogram_ratings.csv",
        help="Path to CSV file with histogram ratings",
    )
    parser.add_argument(
        "--output",
        default="csv_training_results_zscore_extended",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to an existing saved model (e.g. csv_training_results_zscore_extended/best_model.keras)",
    )
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning when resuming",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=30,
        help="Epochs for fine-tuning when resuming",
    )

    args = parser.parse_args()

    try:
        # Load data
        loader = CSVHistogramLoader(args.root_file, args.csv_ratings)
        X, y, names = loader.load_histograms_and_labels()

        # Train model
        trainer = CSVCNNTrainer(args.output)

        if args.resume_from is None:
            results = trainer.train_model(
                X, y, epochs=args.epochs, batch_size=args.batch_size
            )
        else:
            results = trainer.fine_tune(
                X, y,
                model_path=args.resume_from,
                epochs=args.fine_tune_epochs,
                batch_size=args.batch_size,
                learning_rate=args.fine_tune_lr,
            )

        print(f"\nTRAINING COMPLETE!")
        print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        print(f"Training time: {results['training_time']}")
        print(f"Results saved in: {args.output}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
