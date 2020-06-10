import tensorflow as tf
from tensorflow.keras import layers, losses, models, Sequential
import glob
import numpy as np

from preprocessing import preprocess

class MessengerTextGenerator:
    def __init__(self,input_folder):
        """
        Parameters
        ----------
        input_folder : str
            Folder containing .json
            
        """
        
        self.BUFFER_SIZE = 10000
        self.input_folder = input_folder
        self.model = None
        self.dataset = None
        
    def __split_input_target(self,chunk):
        """Take out last char of chunk for target and rest as input."""
    
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    
    def __create_model(self,embedding_dim,units,dropout_rate,batch_size):
        """Create model for training and for generating text (with batch_size = 1)."""
        
        model = Sequential()
        model.add(layers.Embedding(self.nb_char,embedding_dim,
                                        batch_input_shape=[batch_size, None]))
        model.add(layers.LSTM(units, activation='tanh',stateful=True,return_sequences=True))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.nb_char))
        return model
        
    def preprocessing(self, seq_length, embedding_dim, units, dropout_rate, batch_size):
        """
        Parameters
        ----------
        seq_length : int
            Input length in the network.
        embedding_dim : int
            Embedding Dimension.
        units : int
            Dimensionality of the LSTM output space.
        dropout_rate : float
            Dropout rate between 0 and 1.
        batch_size : int
            Batch size.
        """
        
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.units = units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        
        # Extract all sentences from json files
        list_json = glob.glob(self.input_folder+'\*.json')
        dic_senders = preprocess(list_json)
        all_sentences = ''
        for key in dic_senders: # select all senders
            sentences_with_sender = [key+ ' : ' + s + '\n' for s in dic_senders[key]]
            for s in sentences_with_sender:
                all_sentences += s # and add their message together        
        vocab = sorted(set(all_sentences))   
        
        # Creating a mapping from unique characters to indices
        self.__char2idx = {u:i for i, u in enumerate(vocab)}
        self.__idx2char = np.array(vocab)
        self.nb_char = len(vocab)
        print ('{} unique characters'.format(self.nb_char))
        
        # Create dataset for training
        text_as_int = np.array([self.__char2idx[c] for c in all_sentences])
        dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = dataset.batch(self.seq_length+1, drop_remainder=True)
        dataset = sequences.map(self.__split_input_target)
        self.dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.batch_size, drop_remainder=True)
        
        # Create model
        self.model = self.__create_model(embedding_dim,units,dropout_rate,self.batch_size)
        self.model.summary()
        self.model.compile(optimizer='adam',
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
    def train(self, epochs, verbose = 1):
        """ Train the model with previously created dataset.
        
        Parameters
        ----------
        epochs : int
            Number of epochs to train the model.
        verbose : int
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        """
        
        if self.model is None:
            print("Call preprocessing first")
            return
        history = self.model.fit(self.dataset, epochs=epochs, verbose = verbose)
        print("Model trained for {} epochs".format(epochs))
        return history
        
    def generate_text(self,nb_generate = 1000, start_string=u'Guillaume Sachet : '):
        """ Generate characters from network
        
        Parameters
        ----------
        nb_generate : int
            Number of characters to generate.
        start_string : unicode str
            First characters of the generated text. Input to the network.
        """
        
        if self.model is None:
            print("Create a model first")
            return

        model_gen = self.__create_model(self.embedding_dim, self.units, self.dropout_rate,1)
        model_gen.set_weights(self.model.get_weights())
        # Converting our start string to numbers (vectorizing)
        input_eval = [self.__char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Here batch size == 1
        model_gen.reset_states()
        for i in range(nb_generate):
            predictions = model_gen.predict(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.__idx2char[predicted_id])

        return (start_string + ''.join(text_generated))