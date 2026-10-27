import tensorflow as tf
import numpy as np
import pickle
import json
import os
import streamlit as st
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


## Positional Encoding Block
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,D_MODEL):
        super().__init__()
        self.D_MODEL = D_MODEL

    def call(self,x):
        seq_len = tf.shape(x)[1]

        position = tf.range(seq_len)
        position = tf.expand_dims(position,axis=1)
        position = tf.cast(position,tf.float32)

        i = tf.range(0,self.D_MODEL,2)
        i = tf.cast(i,tf.float32)

        div_term = tf.pow(
                   tf.constant(10000.0,dtype=tf.float32),
                    i/tf.cast(self.D_MODEL,tf.float32))

        angle = position/div_term

        sin_values = tf.sin(angle)
        cos_values = tf.cos(angle)

        pe = tf.stack([sin_values,cos_values],axis=2)
        pe = tf.reshape(pe,(seq_len,self.D_MODEL))

        return pe[tf.newaxis, :, :]






## Transformer Encoder Block
class TransformerEncoderBlock(tf.keras.layers.Layer):

    def __init__(self, NUM_HEADS, D_MODEL, KEY_DIM, D_FF):
        super().__init__()

        self.NUM_HEADS = NUM_HEADS
        self.D_MODEL = D_MODEL
        self.D_FF = D_FF
        self.KEY_DIM = KEY_DIM

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.NUM_HEADS,
            key_dim=self.KEY_DIM
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_dense_1 = tf.keras.layers.Dense(self.D_FF)
        self.gelu = tf.keras.layers.Activation("gelu")
        self.ffn_dense_2 = tf.keras.layers.Dense(self.D_MODEL)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, padding_mask=None):

        attention_mask = None

        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask, tf.bool)
            attention_mask = padding_mask[:, tf.newaxis, :]

        
        attention_output = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask
        )

        normalized1 = self.norm1(x + attention_output)

        ffn_hidden = self.ffn_dense_1(normalized1)
        ffn_activated = self.gelu(ffn_hidden)
        ffn_output = self.ffn_dense_2(ffn_activated)

        output = self.norm2(normalized1 + ffn_output)

        return output
















## Transformer Decoder Block
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self,NUM_HEADS,KEY_DIM,D_MODEL,D_FF):
        super().__init__()
        self.NUM_HEADS = NUM_HEADS
        self.KEY_DIM = KEY_DIM
        self.D_MODEL = D_MODEL
        self.D_FF = D_FF

        self.self_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.NUM_HEADS,
            key_dim = self.KEY_DIM
        )

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.cross_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.NUM_HEADS,
            key_dim = self.KEY_DIM
        )

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.ffn_dense_1 = tf.keras.layers.Dense(
            self.D_FF,
            activation=None
        )

        self.gelu = tf.keras.layers.Activation("gelu")

        self.ffn_dense_2 = tf.keras.layers.Dense(
            self.D_MODEL,
            activation=None
        )

        self.norm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self,x,encoder_output,decoder_padding_mask=None,encoder_padding_mask=None,training=None):

        self_attention_mask=None

        if decoder_padding_mask is not None:
            padding_mask = tf.cast(decoder_padding_mask,tf.bool)
            self_attention_mask= padding_mask[:, tf.newaxis, :]

        seq_len = tf.shape(x)[1]

        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len,seq_len),dtype=tf.int32),
            -1,
             0
        )
        causal_mask = tf.cast(causal_mask,tf.bool)
        causal_mask = causal_mask[tf.newaxis, :, :]

        if self_attention_mask is not None:
            
            self_attention_mask = tf.logical_and(
                self_attention_mask,
                causal_mask
            )
        else:
            self_attention_mask = causal_mask


        self_attention_output = self.self_mha(
            query=x,
            key=x,
            value=x,
            attention_mask=self_attention_mask,
            training=training
        )


        normalized1 = self.norm1(x + self_attention_output)


        cross_attention_mask = None

        if encoder_padding_mask is not None:
            encoder_padding_mask = tf.cast(encoder_padding_mask,tf.bool)
            cross_attention_mask= (encoder_padding_mask[:, tf.newaxis, :])

        cross_attention_output = self.cross_mha(
                    query=normalized1,
                    key=encoder_output,
                    value = encoder_output,
                    attention_mask=cross_attention_mask,
                    training=training
                )
        normalized2 = self.norm2(normalized1 + cross_attention_output)
        ffn_hidden = self.ffn_dense_1(normalized2)
        ffn_activated = self.gelu(ffn_hidden)
        ffn_output = self.ffn_dense_2(ffn_activated)

        output = self.norm3(normalized2 + ffn_output)
            
        return output  














## Transformer Block
class TransformerBlock(tf.keras.Model):
    def __init__(self,english_vocab_size,french_vocab_size,max_english_len,max_french_len,D_MODEL,NUM_HEADS,D_FF,dropout=0.1):
        super().__init__()
        self.D_MODEL=D_MODEL
        self.NUM_HEADS=NUM_HEADS
        self.KEY_DIM = D_MODEL//NUM_HEADS
        self.D_FF = D_FF

        self.english_embedding = tf.keras.layers.Embedding(
            input_dim=english_vocab_size,
            output_dim=D_MODEL,
            mask_zero=True
        )

        self.french_embedding = tf.keras.layers.Embedding(
            input_dim=french_vocab_size,
            output_dim=D_MODEL,
            mask_zero=True
        )

        self.positional_encoding = PositionalEncoding(D_MODEL)

        self.dropout= tf.keras.layers.Dropout(dropout)

        self.encoder_block = TransformerEncoderBlock(
            NUM_HEADS=NUM_HEADS,
            D_MODEL = D_MODEL,
            KEY_DIM=D_MODEL//NUM_HEADS,
            D_FF=D_FF
        )
        self.decoder_block = TransformerDecoderBlock(
            NUM_HEADS=NUM_HEADS,
            KEY_DIM=D_MODEL//NUM_HEADS,
            D_MODEL=D_MODEL,
            D_FF=D_FF
        )
        ## Final output layer
        self.output_layer = tf.keras.layers.Dense(
            french_vocab_size
        )

    def call(self,inputs,training=None):
        english_ids,french_ids=inputs
        
        ## Create padding mask
        english_padding_mask=tf.not_equal(
            english_ids,0
        )

        french_padding_mask=tf.not_equal(
            french_ids,0
        )

        encoder_x = self.english_embedding(english_ids)

        encoder_x *= tf.math.sqrt(
            tf.cast(self.D_MODEL,tf.float32)
        ) 

        encoder_x += self.positional_encoding(encoder_x)


        encoder_x = self.dropout(
                encoder_x,
                training=training
        )

        encoder_output = self.encoder_block(
            encoder_x,
            padding_mask=english_padding_mask
        )

        ### DECODER

        decoder_x = self.french_embedding(french_ids)

        decoder_x *= tf.math.sqrt(
            tf.cast(self.D_MODEL,tf.float32)
        )

        decoder_x += self.positional_encoding(decoder_x)

        decoder_x = self.dropout(
            decoder_x,
            training=training
        )

        decoder_output = self.decoder_block(
            x = decoder_x,
            encoder_output = encoder_output,
            decoder_padding_mask=french_padding_mask,
            encoder_padding_mask=english_padding_mask,
            training=training
        )


        output = self.output_layer(decoder_output)

        return output









## Model Loading code

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration
with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config = json.load(f)

# Load vocabularies
with open(os.path.join(BASE_DIR, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

english_token_to_id = vocab["english_token_to_id"]
french_token_to_id = vocab["french_token_to_id"]

english_id_to_token = vocab["english_id_to_token"]
french_id_to_token = vocab["french_id_to_token"]

# Load model using the SAME TransformerBlock class
# and the SAME constructor arguments as in your notebook.

model = TransformerBlock(
    # Add your exact training-time constructor arguments here.
     english_vocab_size=len(english_token_to_id),
    french_vocab_size=len(french_token_to_id),
    max_english_len=config["MAX_ENGLISH_LEN"],
    max_french_len=config["MAX_FRENCH_LEN"],
    D_MODEL=config["D_MODEL"],
    NUM_HEADS=config["NUM_HEADS"],
    D_FF=config["D_FF"],
    dropout=config["DROPOUT"]
    
)

# Build the model before loading weights.
# Use the same input structure as model.fit().
dummy_english = tf.zeros(
    (1, config["MAX_ENGLISH_LEN"]),
    dtype=tf.int32
)

dummy_decoder = tf.zeros(
    (1, config["MAX_FRENCH_LEN"] - 1),
    dtype=tf.int32
)

_ = model((dummy_english, dummy_decoder))

# Load trained weights
weights_path = os.path.join(
    BASE_DIR,
    "transformer_translation.weights.h5"
)
print("BASE_DIR:", BASE_DIR)
print("Weights path:", weights_path)
print("File exists:", os.path.exists(weights_path))

model.load_weights(weights_path)
model.load_weights(weights_path)

print("Transformer model loaded successfully!")
















def tokenize(sentence):
    sentence = sentence.lower().strip()

    # Add a period if the input has no ending punctuation
    if sentence and sentence[-1] not in ".!?":
        sentence += "."

    return re.findall(r"\w+|[^\w\s]", sentence)


def translate_sentence(sentence):

    # 1. Tokenize the English sentence
    english_tokens = tokenize(sentence)

   
    # 2. Convert words into English token IDs
    english_ids = [
        english_token_to_id.get(
            token,
            english_token_to_id["<UNK>"]
        )
        for token in english_tokens
    ]

    # 3. Pad the English input
    english_padded = pad_sequences(
        [english_ids],
        maxlen=config["MAX_ENGLISH_LEN"],
        padding="post",
        truncating="post",
        value=english_token_to_id["<PAD>"],
        dtype="int32"
    )

    encoder_input = tf.constant(
        english_padded,
        dtype=tf.int32
    )

    # 4. Get French special-token IDs
    BOS_ID = french_token_to_id["<BOS>"]
    EOS_ID = french_token_to_id["<EOS>"]
    PAD_ID = french_token_to_id["<PAD>"]

    # 5. Start decoding with BOS
    decoder_ids = [BOS_ID]

    # 6. Generate French tokens
    max_french_len = config["MAX_FRENCH_LEN"]

    for _ in range(max_french_len - 1):

        decoder_input = tf.constant(
            [decoder_ids],
            dtype=tf.int32
        )

        predictions = model(
            (encoder_input, decoder_input),
            training=False
        )

        next_token_id = int(
            tf.argmax(
                predictions[0, -1, :],
                axis=-1
            ).numpy()
        )

        # Stop at EOS or PAD
        if next_token_id == EOS_ID or next_token_id == PAD_ID:
            break

        decoder_ids.append(next_token_id)

    # 7. Convert IDs into French tokens
    french_tokens = [
        french_id_to_token.get(token_id, "<UNK>")
        for token_id in decoder_ids[1:]
    ]

    # 8. Join tokens and clean punctuation
    translation = " ".join(french_tokens)

    translation = re.sub(
        r"\s+([?.!,;:])",
        r"\1",
        translation
    )

    translation = re.sub(
        r"\s+-\s+",
        "-",
        translation
    )

    return translation
