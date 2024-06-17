import os
import music21
import json
import numpy as np
import keras

#dataset path
KERN_DATASET_PATH = './essen/europa/deutschl/erk'
SAVE_DIR = './dataset'
MAPPING_PATH = './mapping.json'

SEQUENCE_LENGTH = 64

ACCEPTABLE_DURATIONS = [
    0.25,
    0.50,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

def load_music_in_kern(dataset_path):
    
    songs = []

    #go through all the folders and files
    for path, subdirs, files in os.walk(dataset_path):
        
        for file in files:
            if file[-3:] == 'krn':
                song = music21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs

def has_acceptable_duration(song, acceptable_durations):

    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False

    return True

def transpose(song):

    #get the key from the score
    part = song.getElementsByClass(music21.stream.Part)
    measures_part0 = part[0].getElementsByClass(music21.stream.Measure)
    key = measures_part0[0][4]

    #estimate the key using music21
    if not isinstance(key, music21.key.Key):
        key = song.analyze("key")


    #get interval for transposition eg. Bmaj -> Cmaj
    if key.mode == "major":
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch("A"))

    #transpose the song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song, time_step=0.25):
    encoded_song = []
    #accept song in music21 representation
    #return it in time series representation
    for event in song.flat.notesAndRests:

        #handle notes
        if isinstance(event, music21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, music21.note.Rest):
            symbol = 'r'
        
        #convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
        
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def load(file_path):
    song = ""
    with open (file_path, "r") as fp:
        song = fp.read()

    return song

def create_single_song_database(dataset_path, out_dataset_path, sequence_length):

    new_song_delimiter = "/ " * sequence_length

    songs = ""

    #get all the songs in the dataset path
    for path, _, files in os.walk(dataset_path):
        
        for file in files:
            file_path = os.path.join(path, file)
            #string saved in the file_path
            song = load(file_path)
            songs += song + " " + new_song_delimiter

    #save the songs to the out_dataset_path
    with open(out_dataset_path, "w") as fp:
        fp.write(songs)
    
    #return
    return songs

#preprocess function
def preprocess(dataset_path):

    #loading the songs
    print("Loading songs...")
    songs = load_music_in_kern(KERN_DATASET_PATH)
    print(f'Loaded {len(songs)} songs')

    #filter out all the songs that have inacceptable durations
    for i, song in enumerate(songs):
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue

        #transpose the songs to Cmaj/Amin
        song = transpose(song)

        #encode the song
        encoded_song = encode_song(song)

        #save the song
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

def create_mapping(songs, mapping_path):
    
    #identify the vocabulary
    songs = songs.split()
    vocabulary = sorted(list(set(songs)))

    #create the mapping
    mappings = {}
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    #save vocabulary to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_song_to_int(songs):
    int_songs = []
    
    #load mappings
    with open("./mapping.json", "r") as fp:
        mappings = json.load(fp)
    
    #cast songs string to list
    songs = songs.split()

    #map
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs

def generate_train_sequences(sequence_length):

    #load songs and map them to int
    songs = load("./final_dataset/final_file")
    int_songs = convert_song_to_int(songs)

    #generate training symbols
    #100 symbols, 64 sequence_len, 36 training_samples    
    num_sequences = len(int_songs) - sequence_length

    inputs = []
    targets = []
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    #one-hot encode the sequences
    #inputs (# of sequences, sequence_length, vocabulary_size)
    #targets (# of sequences, vacabulary_size)

    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs,
                            num_classes=vocabulary_size)
    
    targets = np.array(targets)

    return inputs, targets


    return inputs, targets
def main():
    preprocess(KERN_DATASET_PATH)
    
    songs = create_single_song_database(SAVE_DIR, os.path.join("./final_dataset", str("final_file")), SEQUENCE_LENGTH)

    create_mapping(songs, MAPPING_PATH)

    inputs, targets = generate_train_sequences(SEQUENCE_LENGTH)



if __name__ == "__main__":
    main()