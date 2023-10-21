from nptdms import TdmsFile
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

file_name = "RandomEverything1.5mins.tdms"
#file_name = "HoldPos.tdms"
#file_name = "ContinuousChangeDir.tdms"
#file_name = "MoveHoldMove.tdms"

def analog_to_digital(v, threshold=2.5):
    if v > threshold:
        return 1
    else:
        return 0
    


#Create sequences of the desired length
# def create_sequences(data, time_steps):
#     sequences = []
#     for i in range(len(data) - time_steps + 1):
#         sequence = data[i : i + time_steps]
#         sequences.append(sequence)
#     return np.array(sequences)


def create_sequences(data, labels, time_steps):
    sequences = []
    labels_list = []
    #print(len(data))
    for i in range(len(data) - time_steps + 1):
      
        sequence = data[i : i + time_steps]
        label = labels[i : i + time_steps]#[i + time_steps - 1]  # Assuming labels correspond to the last time step
        sequences.append(sequence)
        labels_list.append(label)

    return np.array(sequences), np.array(labels_list)


with TdmsFile.open(file_name) as tdms_file:

    time_steps = 150

    group = tdms_file['Log']
    print(group.channels())
    A1 = group["A1"][:]
    o_len = len(A1) - len(A1)%time_steps
    A1 = A1[:o_len]
    B1 = group["A2"][:o_len]



    Step = group["Step"][:o_len]
    Dir = group["Direction"][:o_len]


    for i in range(len(Step)):
        Step[i] = analog_to_digital(Step[i])

    for i in range(len(Dir)):
        Dir[i] = analog_to_digital(Dir[i])


    

    # plt.plot(A1)
    # plt.plot(B1)
    # plt.plot(Step)
    # plt.plot(Dir)
    # plt.show()

    
    
    features = 2
    # ok = False
    # if ok:
    input_A = tf.keras.layers.Input(shape=(time_steps, 1))
    input_B = tf.keras.layers.Input(shape=(time_steps, 1))

    branch_A = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=False, name='lstm_A2'),
       # tf.keras.layers.Flatten(name='flatten_A')
        # Add more layers as needed
    ])(input_A)
    
    branch_B = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=False, name='lstm_B2'),
       # tf.keras.layers.Flatten(name='flatten_B')
        # Add more layers as needed
    ])(input_B)

    merged = tf.keras.layers.Concatenate()([branch_A, branch_B])

    output_step = tf.keras.layers.Dense(1, name='output_step')(merged)  # Output for step logic
    #output_direction = tf.keras.layers.Dense(1, name='output_direction')(merged)  # Output for direction logic

    #model = tf.keras.Model(inputs=[input_A, input_B], outputs=[output_step, output_direction])
    model = tf.keras.Model(inputs=[input_A, input_B], outputs=[output_step])

    #model = tf.keras.models.load_model("my_model")
    #model.compile(optimizer='adam', loss={'output_step': 'binary_crossentropy', 'output_direction': 'binary_crossentropy'}, metrics={'output_step': 'accuracy', 'output_direction': 'accuracy'})
    model.compile(optimizer='adam', loss={'output_step': 'binary_crossentropy'}, metrics={'output_step': 'accuracy'})
# Train the model using model.fit with the appropriate training data

    # Combine coil_A and coil_B data into a single input array
   # print(len(A1), len(B1))

    #combined_input = np.stack((A1, B1), axis=-1)
    #labels = Dir#np.stack((Step, Dir), axis=-1)#(Step, Dir)

    

    #dataset = tf.data.Dataset.from_tensor_slices((A1, B1, Step, Dir))

    # coilA_seq = create_sequences(A1, time_steps)
    # coilB_seq = create_sequences(B1, time_steps)
    # step_seq = create_sequences(Step, time_steps)
    # dir_seq = create_sequences(Dir, time_steps)

    d_a, d_b = create_sequences(A1, B1, time_steps)
    l_s, l_d = create_sequences(Step, Dir, time_steps)


    #combined_input = np.stack((coilA_seq, coilB_seq), axis=-1) #np.hstack((A1, B1))#np.stack((A1, B1), axis=-1)

   # print(combined_input)

    # Define the target labels (step and direction)
    #target_labels = np.stack((step_seq, dir_seq), axis=-1) #np.hstack((Step, Dir))#np.stack((Step, Dir), axis=-1)

    # Split the data into training and validation sets
    split_ratio = 0.8  # 80% for training, 20% for validation
    split_index = int(len(d_a) * split_ratio)

    train_data_A = d_a[:split_index]
    train_data_B = d_b[:split_index]
    train_labels_step = l_s[:split_index]
    train_labels_direction = l_d[:split_index]
    #assert(len(train_labels)==len(train_data))
    #assert(len(train_data)==split_index)

    val_data_A = d_a[split_index:]
    val_data_B = d_b[split_index:]
    val_labels_step = l_s[split_index:]
    val_labels_direction = l_d[split_index:]

    #print(len(val_data), len(val_labels))

    # Train the model
    #model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=32)

    model.fit(
        {'input_1': train_data_A, 'input_2': train_data_B},
        #{'output_step': train_labels_step, 'output_direction': train_labels_direction},
        {'output_step': train_labels_step},
        validation_data=({'input_1': val_data_A, 'input_2': val_data_B}, {'output_step': val_labels_step}),
        epochs=2, batch_size=32
    )

    # Evaluate the model
    #loss, accuracy = model.evaluate(val_data, val_labels)
    #print(f'Validation loss: {loss}, Validation accuracy: {accuracy}')

    # Make predictions
    # predictions = model.predict({'input_1': val_data_A, 'input_2': val_data_B})
    # print(predictions, len(A1))
    # print(len(predictions[0]))

    # v = []
    # p = []
    # for i in range(len(predictions[0])):
    #     p.append(predictions[0][i][99])
    #     v.append(val_labels_step[i][99])

    # plt.plot(p)
    # plt.plot(v)
    # plt.show()
    model.save("my_model3", save_format="tf")