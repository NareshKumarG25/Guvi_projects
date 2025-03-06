import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        # Ensure decoder_hidden is reshaped correctly to [batch_size, hidden_size]
        decoder_hidden = decoder_hidden.squeeze(0)  # Remove the first dimension (1, batch_size, hidden_size)
        
        # Compute attention scores by applying the attention mechanism
        attn_weights = torch.matmul(encoder_outputs, decoder_hidden.unsqueeze(2))  # [batch_size, seq_len, 1]
        attn_weights = attn_weights.squeeze(2)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_weights, dim=1)  # Softmax over seq_len dimension
        
        # Compute the context vector as a weighted sum of encoder outputs
        context_vector = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=1)  # [batch_size, hidden_size]
        
        return context_vector, attn_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).unsqueeze(1)  # Add batch dimension
        context, _ = self.attention(hidden[0], encoder_outputs)
        rnn_input = embedded + context.unsqueeze(1)
        output, (hidden, cell) = self.rnn(rnn_input, hidden)
        output = self.fc_out(output.squeeze(1))
        return output, (hidden, cell)

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.output_size = output_size  # Save output_size as a class attribute

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        seq_len = tgt.size(1)
        
        # Use self.output_size here instead of the undefined output_size
        output = torch.zeros(batch_size, seq_len, self.output_size).to(device)

        # Encoder forward pass
        encoder_outputs, (hidden, cell) = self.encoder(src)

        # First token is fed to decoder as the initial input
        decoder_input = tgt[:, 0]
        
        for t in range(1, seq_len):
            # Decoder forward pass with attention mechanism
            output_t, (hidden, cell) = self.decoder(decoder_input, (hidden, cell), encoder_outputs)

            # Store output
            output[:, t] = output_t

            # Use teacher forcing or model's own predictions as the next input
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output_t.argmax(1)  # Get the token with highest probability

            decoder_input = tgt[:, t] if teacher_force else top1

        return output