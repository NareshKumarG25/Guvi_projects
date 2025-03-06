import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
import matplotlib.pyplot as plt

# Generate synthetic data (source and target sequences)
def generate_data(num_samples, seq_length):
    X = []
    y = []
    for _ in range(num_samples):
        seq = np.random.randint(1, 100, seq_length)  # Random integers between 1 and 100
        X.append(seq)
        y.append(seq[::-1])  # Reverse the sequence for target
    return np.array(X), np.array(y)

# Example: Generate a dataset with 1000 samples, each of length 10
X_train, y_train = generate_data(1000, 10)
X_test, y_test = generate_data(200, 10)

# Seq2Seq without attention (Simple LSTM Model)
def create_seq2seq_model_no_attention(input_seq_len, output_seq_len, hidden_units=256):
    encoder_inputs = Input(shape=(input_seq_len,))
    encoder_embedding = Embedding(input_dim=100, output_dim=hidden_units)(encoder_inputs)
    encoder_lstm, forward_h, forward_c = LSTM(hidden_units, return_state=True)(encoder_embedding)
    
    decoder_inputs = Input(shape=(output_seq_len,))
    decoder_embedding = Embedding(input_dim=100, output_dim=hidden_units)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_sequences=True)(decoder_embedding, initial_state=[forward_h, forward_c])
    output = Dense(100, activation='softmax')(decoder_lstm)
    
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Seq2Seq with Attention
def create_seq2seq_model_with_attention(input_seq_len, output_seq_len, hidden_units=256):
    encoder_inputs = Input(shape=(input_seq_len,))
    encoder_embedding = Embedding(input_dim=100, output_dim=hidden_units)(encoder_inputs)
    encoder_lstm, forward_h, forward_c = LSTM(hidden_units, return_state=True, return_sequences=True)(encoder_embedding)
    
    decoder_inputs = Input(shape=(output_seq_len,))
    decoder_embedding = Embedding(input_dim=100, output_dim=hidden_units)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_state=True, return_sequences=True)
    decoder_lstm_out, _, _ = decoder_lstm(decoder_embedding, initial_state=[forward_h, forward_c])
    
    # Attention Mechanism
    attention_layer = Attention()
    attention_output = attention_layer([decoder_lstm_out, encoder_lstm])
    
    concat_layer = Concatenate(axis=-1)([decoder_lstm_out, attention_output])
    output = Dense(100, activation='softmax')(concat_layer)
    
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Set parameters
input_seq_len = 10
output_seq_len = 10
hidden_units = 256

# Create both models
model_no_attention = create_seq2seq_model_no_attention(input_seq_len, output_seq_len, hidden_units)
model_with_attention = create_seq2seq_model_with_attention(input_seq_len, output_seq_len, hidden_units)

# Train the models
history_no_attention = model_no_attention.fit([X_train, y_train], y_train, epochs=10, validation_split=0.2)
history_with_attention = model_with_attention.fit([X_train, y_train], y_train, epochs=10, validation_split=0.2)

# Plot loss curves for both models
plt.figure(figsize=(12, 6))
plt.plot(history_no_attention.history['loss'], label='Seq2Seq without Attention - Train Loss')
plt.plot(history_no_attention.history['val_loss'], label='Seq2Seq without Attention - Val Loss')
plt.plot(history_with_attention.history['loss'], label='Seq2Seq with Attention - Train Loss')
plt.plot(history_with_attention.history['val_loss'], label='Seq2Seq with Attention - Val Loss')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate both models on test data
loss_no_attention, accuracy_no_attention = model_no_attention.evaluate([X_test, y_test], y_test)
loss_with_attention, accuracy_with_attention = model_with_attention.evaluate([X_test, y_test], y_test)

print(f"Seq2Seq Model without Attention - Test Loss: {loss_no_attention}, Test Accuracy: {accuracy_no_attention}")
print(f"Seq2Seq Model with Attention - Test Loss: {loss_with_attention}, Test Accuracy: {accuracy_with_attention}")
