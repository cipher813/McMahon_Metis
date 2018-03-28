from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, TensorBoard

# TODO add functionality for partially trained model

first_layer = 1024
drop = 0.3

def create_network(network_input, n_vocab, weight_file=None):
    print("\n**LSTM model initializing**")
    # network input shape (notes - sequence length, sequence_length, 1)
    timesteps = network_input.shape[1] # sequence length
    data_dim = network_input.shape[2] # 1

    print("Input nodes: {} Dropout: {}".format(first_layer, drop))
    print("Input shape (timesteps, data_dim): ({},{})".format(timesteps, data_dim))
    # for LSTM models, return_sequences sb True for all but the last LSTM layer
    # this will input the full sequence rather than a single value
    model = Sequential()
    model.add(LSTM(first_layer, input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(first_layer, return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(first_layer, return_sequences=True)) # added new layer
    model.add(Dropout(drop)) # added new layer
    model.add(LSTM(first_layer))
    model.add(Dense(first_layer//2))
    model.add(Dropout(drop))
    model.add(Dense(n_vocab)) # based on number of unique system outputs
    model.add(Activation('softmax'))

    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay = 0.0)
    model.compile(loss='categorical_crossentropy',optimizer=rms)

    # this is a complete model file
    if weight_file:
        weights = model.load_weights(weight_file)
        print("LSTM modelinitialized for midi CREATION with model from {}".format(weight_file))
    else:
        print("LSTM model initialized for TRAINING - model being generated (no weights file)")
    return model


# def load_saved_model(model_file):
#     model = load_model(model_file)
#     return model


def train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length):
    # saves model after each epoch
    check_stats = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}-'
    weight_file = output_tag + check_stats + 'weights.hdf5'
    model_checkpoint = weight_file
    checkpoint = ModelCheckpoint(model_checkpoint,
                                    monitor='loss',
                                    verbose=0,
                                    save_best_only=False,
                                    mode='min',
                                    period=1)
    callbacks_list = [checkpoint]

    tensorboard = TensorBoard(log_dir='../log',
                                histogram_freq=1,
                                batch_size=batch_size,
                                write_graph=True,
                                write_grads=True,
                                write_images=True,
                                embeddings_freq=1)

    print("Fitting Model. \nNetwork Input Shape: {} Network Output Shape: {}".format(network_input_r.shape,network_output_r.shape))
    print("Epochs: {} Batch Size: {}".format(epochs, batch_size))
    model.fit(network_input_r, network_output_r, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    # saves model upon training completion
    # weight_file = output_tag + 'lstm-weights.hdf5'
    model.save_weights(weight_file)
    print("TRAINING complete - weights saved at: {}".format(weight_file))
    return model, weight_file
