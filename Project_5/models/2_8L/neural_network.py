from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Dense, Dropout, LSTM, Activation, GRU, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

first_layer = 512
drop = 0.4

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
    model.add(Bidirectional(LSTM(first_layer, dropout=drop, recurrent_dropout=drop, return_sequences=True), input_shape=(timesteps, data_dim))) # , return_sequences=True
    model.add(LSTM(first_layer, dropout=drop, recurrent_dropout=drop))
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


def train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length):
    # saves model after each epoch
    # check_stats = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}-'
    # weight_file = output_tag + check_stats + 'weights.hdf5'
    base_tag = output_tag + 'weight-'
    epoch_metrics = '{epoch:02d}-{loss:.4f}-{val_loss:.4f}'
    end_tag = '.hdf5'
    weight_checkpoint = base_tag + epoch_metrics + end_tag
    checkpoint = ModelCheckpoint(weight_checkpoint,
                                    monitor='loss',
                                    verbose=0,
                                    save_best_only=False,
                                    mode='min',
                                    period=1)

    # https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    tensorboard = TensorBoard(log_dir='log', histogram_freq=1, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, tensorboard]

    print("Fitting Model. \nNetwork Input Shape: {} Network Output Shape: {}".format(network_input_r.shape,network_output_r.shape))
    print("Epochs: {} Batch Size: {}".format(epochs, batch_size))
    model.fit(network_input_r, network_output_r, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.2)

    # saves model upon training completion
    weight_file = output_tag + 'final_weights.hdf5'
    model.save_weights(weight_file)
    print("TRAINING complete - weights saved at: {}".format(weight_file))
    return model, weight_file
