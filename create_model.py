import tensorflow as tf


def create_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

    opt = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
