import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
