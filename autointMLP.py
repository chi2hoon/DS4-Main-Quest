# -*- coding: utf-8 -*-
"""
AutoIntMLP (AutoInt+) TensorFlow 2.x 구현
------------------------------------------------
기존 AutoInt(Self-Attention 기반 Feature Interaction) 결과와
DNN(MLP) 결과를 결합해 최종 로짓을 계산합니다.

• AutoIntMLP  : tf.keras.layers.Layer
• AutoIntMLPModel : tf.keras.models.Model 래퍼

본 파일은 학습/저장을 위한 예시 코드까지 포함하므로
별도의 스크립트 없이 import 후 즉시 사용할 수 있습니다.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Layer
from tensorflow.keras.models import Model

# 기존 AutoInt 모듈에서 재사용 가능한 레이어를 가져옵니다.
# 패키지 구조(autoint 디렉터리)에 따라 두 가지 경로를 모두 시도합니다.
try:
    from autoint.autoint import (
        FeaturesEmbedding,
        MultiHeadSelfAttention,
        MultiLayerPerceptron,
    )
except ImportError:
    # fallback: autoint.py 를 직접 모듈로 import
    try:
        from autoint import (
            FeaturesEmbedding,
            MultiHeadSelfAttention,
            MultiLayerPerceptron,
        )
    except ImportError as e:
        raise ImportError(
            "AutoInt 관련 레이어를 찾지 못했습니다. autoint 디렉터리 안에 autoint.py가 존재해야 합니다."
        ) from e


class AutoIntMLP(Layer):
    """AutoInt + MLP layer (Keras)"""

    def __init__(
        self,
        field_dims: Union[np.ndarray, Sequence[int]],
        embedding_size: int = 16,
        att_layer_num: int = 3,
        att_head_num: int = 2,
        att_res: bool = True,
        dnn_hidden_units: Tuple[int, ...] = (128, 64),
        dnn_activation: str = "relu",
        l2_reg_dnn: float = 0.0,
        dnn_use_bn: bool = False,
        dnn_dropout: float = 0.4,
        init_std: float = 0.0001,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size

        # Embedding
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)

        # Self-Attention (AutoInt) 파트
        self.att_layers = [
            MultiHeadSelfAttention(
                att_embedding_size=embedding_size,
                head_num=att_head_num,
                use_res=att_res,
            )
            for _ in range(att_layer_num)
        ]
        self.att_dense = Dense(
            1,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=init_std),
        )

        # MLP 파트
        self.mlp = MultiLayerPerceptron(
            input_dim=self.num_fields * embedding_size,
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            output_layer=False,
            init_std=init_std,
        )
        self.mlp_dense = Dense(
            1,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=init_std),
        )

    def call(self, inputs, training: bool = False):
        # Embedding Lookup  (batch, num_fields, embedding_size)
        embed_x = self.embedding(inputs)

        # AutoInt 경로
        att_out = embed_x
        for layer in self.att_layers:
            att_out = layer(att_out)
        att_out_flat = Flatten()(att_out)
        att_logit = self.att_dense(att_out_flat)

        # MLP 경로
        mlp_inp = tf.reshape(embed_x, (-1, self.num_fields * self.embedding_size))
        mlp_hidden = self.mlp(mlp_inp, training=training)
        mlp_logit = self.mlp_dense(mlp_hidden)

        # 로짓 합산 후 sigmoid
        y_pred = tf.nn.sigmoid(att_logit + mlp_logit)
        return y_pred


class AutoIntMLPModel(Model):
    """Keras Model wrapper for AutoIntMLP"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.core = AutoIntMLP(*args, **kwargs)

    def call(self, inputs, training: bool = False):  # type: ignore[override]
        return self.core(inputs, training=training)


# ──────────────────────────────────────────────────────────────
# 학습/저장 예시 스크립트 (이 모듈을 직접 실행할 때 동작)
# 해당 부분은 필요 없으면 주석 처리해도 무방합니다.
# ──────────────────────────────────────────────────────────────

def _example_train():  # pragma: no cover
    """간단한 학습 예시 (더미 데이터)"""

    num_samples = 1000
    num_fields = 10
    field_dims = np.random.randint(10, 50, size=num_fields)
    X_dummy = np.random.randint(0, field_dims.max(), size=(num_samples, num_fields))
    y_dummy = np.random.randint(0, 2, size=(num_samples, 1))

    model = AutoIntMLPModel(
        field_dims=field_dims,
        embedding_size=16,
        att_layer_num=3,
        att_head_num=2,
        dnn_hidden_units=(128, 64),
        dnn_dropout=0.3,
    )

    # Build 모델 (더미 입력)
    model(tf.constant(X_dummy[:1], dtype=tf.int64))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    model.fit(X_dummy, y_dummy, epochs=2, batch_size=32)

    # 가중치 저장
    model.save_weights("autoIntMLP_model.weights.h5")
    print("가중치 저장 완료 -> autoIntMLP_model.weights.h5")


if __name__ == "__main__":  # pragma: no cover
    _example_train()