import tensorflow as tf
import keras
import json
import numpy as np
import music21 as m21
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:

    def __init__(self, model_path="./model_trained.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path)
        # batch_size = 1
        # self.model.input.reshape((batch_size,) + self.model.input.shape[1:])
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH

        pass

    def _sample_with_temprature(self, probabilities, temprature):
        
        # temprature -> infinity; homogenous distribution
        # temprature -> 0; highest probability gets picked
        # temprature -> 1;

        predictions = np.log(probabilities) / temprature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

        pass

    def generate_melody(self, seed, num_steps, max_sequence_len, temprature):

        #create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        
        #map seeds to ints
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            #limit the seed to the max_sequence_len
            seed = seed[-max_sequence_len:]

            #one-hot encode the seed
            # (max_sequence_length, len(self._mappings))
            one_hot_seed = keras.utils.to_categorical(seed,
                                    num_classes=len(self._mappings))
            one_hot_seed = one_hot_seed[np.newaxis, ...]

            #make a prediction
            probabilities = self.model.predict(one_hot_seed)[0]
            output_int = self._sample_with_temprature(probabilities, temprature)

            #update seed
            seed.append(output_int)

            output_symbol = [k for k, v in self._mappings.items() if v == output_int]

            #check if we're at the end of the melody
            if output_symbol[0] == '/':
                break

            #update the melody
            melody.append(str(output_symbol[0]))
        
        return melody

    def save_melody(self, melody, format="midi", file_name="mel.midi", step_duration=0.25):

        #create a music21 stream
        stream = m21.stream.Stream()
        
        #parse all the symbols in the melody
        #and create notes and rests
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with notes/rests beyond the first
                if start_symbol is not None:
                    
                    quarter_length_duration = step_duration * step_counter

                    #handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(pitch=int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    #reset the step
                    step_counter = 1
                
                start_symbol = symbol

            else:
                step_counter += 1
    
        #write to a file
        stream.write(format, file_name)



if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ _ 67 _ _ 64 _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 2.0)
    print(melody)

    #save the melody
    mg.save_melody(melody)






