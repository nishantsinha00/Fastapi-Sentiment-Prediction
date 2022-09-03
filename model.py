try:
    import tensorflow as tf
except:
    print("Error importing packages!")
    
class GRU_model(tf.keras.Model):
    
    def __init__(self, max_seq_length, embedding_dim, vocab_length):
        super().__init__()
        self.inp = tf.keras.layers.InputLayer(input_shape=(max_seq_length))
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_length, output_dim = embedding_dim, input_length = max_seq_length)
        self.gru = tf.keras.layers.GRU(units = embedding_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_o = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        inputs = self.inp(inputs)
        embedding = self.embedding(inputs)
        flatten = self.flatten(embedding)
        gru = self.gru(embedding)
        gru_flatten = self.flatten(gru)
        concat = tf.keras.layers.concatenate([flatten, gru_flatten])
        outputs = self.dense_o(concat)
        return outputs
    