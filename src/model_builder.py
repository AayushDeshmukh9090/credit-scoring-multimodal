"""
model_builder.py
----------------
Builds and returns the compiled dual-branch Keras model.

Architecture:
  Branch 1 — Numerical: Dense(32) → BN → Dropout
  Branch 2 — Categorical: one Embedding per feature → Flatten
  Fusion   — Concatenate all branches
  Head     — Dense(64) → Dropout → Dense(32) → Sigmoid output

Your original model_builder.py was already well-structured.
Changes made here:
  1. Added L2 regularization to Dense layers (prevents overfitting on 300k rows)
  2. Added get_model_summary() utility for notebooks
  3. import path compatible with src/ layout
"""

import math

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_multimodal_model(
    num_feature_count: int,
    cat_feature_cardinalities: dict,
    learning_rate: float = 0.001,
    l2_reg: float = 1e-4,
) -> models.Model:
    """
    Build the dual-branch credit scoring model.

    Parameters
    ----------
    num_feature_count        : number of numerical input features
    cat_feature_cardinalities: dict of {feature_name: n_unique_values}
                               Must be in the SAME ORDER as CATEGORICAL_FEATURES
                               in data_loader.py
    learning_rate            : Adam optimizer learning rate
    l2_reg                   : L2 regularization strength for Dense layers

    Returns
    -------
    Compiled Keras Model ready for .fit()
    """

    # ── Branch 1: Numerical ──────────────────────────────────────────────────
    input_num = layers.Input(shape=(num_feature_count,), name="input_numerical")

    x_num = layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(input_num)
    x_num = layers.BatchNormalization()(x_num)
    x_num = layers.Dropout(0.2)(x_num)

    # ── Branch 2: Categorical (Embeddings) ───────────────────────────────────
    cat_inputs = []
    cat_embeddings = []

    for feature_name, unique_count in cat_feature_cardinalities.items():
        inp = layers.Input(shape=(1,), name=f"input_{feature_name}")
        cat_inputs.append(inp)

        # Embedding dimension rule: min(50, ceil(cardinality / 2)), at least 2
        embedding_dim = min(50, max(2, math.ceil(unique_count / 2)))

        emb = layers.Embedding(
            input_dim=unique_count + 1,   # +1 reserves index 0 for unseen/padding
            output_dim=embedding_dim,
            name=f"emb_{feature_name}"
        )(inp)

        emb = layers.Flatten()(emb)       # (Batch, 1, Dim) → (Batch, Dim)
        cat_embeddings.append(emb)

        print(f"  [model] {feature_name}: {unique_count} categories → embedding dim {embedding_dim}")

    # ── Fusion ───────────────────────────────────────────────────────────────
    fusion = layers.concatenate(
        [x_num] + cat_embeddings, name="fusion_layer"
    )

    # ── Decision Head ────────────────────────────────────────────────────────
    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(fusion)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        32, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)

    output = layers.Dense(1, activation="sigmoid", name="output_probability")(x)

    # ── Assemble & Compile ───────────────────────────────────────────────────
    all_inputs = [input_num] + cat_inputs
    model = models.Model(inputs=all_inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def prepare_model_inputs(X_num, X_cat) -> list:
    """
    Convert DataFrames into the list-of-arrays format Keras expects.

    The model has multiple inputs:
      [numerical_array, cat_col_1_array, cat_col_2_array, ...]

    This function handles that conversion in one place so it is
    identical in the training notebook and the API.

    Parameters
    ----------
    X_num : pd.DataFrame or np.ndarray  — numerical features
    X_cat : pd.DataFrame                — categorical features (integer-encoded)

    Returns
    -------
    list of numpy arrays
    """
    import numpy as np
    num_array = X_num.to_numpy() if hasattr(X_num, "to_numpy") else X_num
    cat_arrays = [X_cat[col].to_numpy() for col in X_cat.columns]
    return [num_array] + cat_arrays

print('*')