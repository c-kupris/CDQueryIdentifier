"""Make an ML model that
can identify and return
conserved protein domains
on cell surfaces, that
could focus research on
 chemotherapeutic or
 immunotherapeutic targets."""
import os

import numpy
# Import the required libraries.
from datasets import load_dataset
# import fsspec
import keras  # Uses version 2.15.0
# model_history.values()from matplotlib import pyplot  # Uses version 3.8.1
import pymolviz  # Uses version 1.2.2.1
import pypdb  # Uses version 2.4
import scipy  # Uses version 1.10.1
from sklearn import preprocessing


# Uses tensorflow 2.13.0


# Define the class.
class CDQueryIdentifier(keras.Sequential):
    protein_function_dataset = load_dataset("dylanchia111/ProteinFunctionSequence", split="train")

    train_x_unprocessed_data = protein_function_dataset["text"]

    scikit_learn_one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, sparse_output=False)

    """The scikit-learn OneHotEncoder expects data it is fit to, to be passed to it as a 2-D array. The 

    dylanchia111/ProteinFunctionSequence HuggingFace training dataset is a 1-D array, so we need to 

    convert it to a 2-D array. Data in the dylanchia111/ProteinFunctionSequence HuggingFace training dataset 

    can be separated into two columns: 

    - One for the associated protein's function

    - One for the associated protein's protein-coding sequence

    Once the 1-D dylanchia111/ProteinFunctionSequence HuggingFace training dataset is separated into these two 

    columns, and converted to a new dataset, *this* new dataset is what should be used to fit the Scikit-learn 

    OneHotEncoder for processing. Then, once the data is processed, it can be used to fit the created GAN model."""

    seq = 'SEQUENCE'

    protein_functions = []

    protein_sequences = []

    for text in train_x_unprocessed_data:
        protein_function, seq_separator, protein_sequence = text.partition(seq)
        protein_functions.append(protein_function)
        protein_sequences.append(protein_sequence)

    x_train = numpy.array((numpy.array(protein_functions), numpy.array(protein_sequences)))

    # print("The shape of the training data is: " + str(x_train.shape))

    one_hot_encoded_training_data = scikit_learn_one_hot_encoder.fit_transform(x_train)

    print("one-hot-encoded training data shape: " + str(one_hot_encoded_training_data.shape))

    reshape_input_data = keras.layers.Reshape(target_shape=(10, 24022,))

    reshaped_one_hot_encoded_training_data = reshape_input_data(one_hot_encoded_training_data)

    reshaped_one_hot_encoded_training_data = numpy.expand_dims(reshaped_one_hot_encoded_training_data, axis=0)

    batch_size = 1
    epochs = 20

    # Initialize the class.
    def __init__(self):
        super(CDQueryIdentifier, self).__init__()

    @classmethod
    def get_protein_three_dimensional_structure(cls, protein_id: str):
        """Given a protein of interest, returns a volumetric view of the protein."""
        protein_id = CDQueryIdentifier.pdb_search(protein_id)
        protein_volume = pymolviz.Volume(grid_data=scipy.interpolate.GridData([(0, 0), (1, 1)]),
                                         name=protein_id)
        return protein_volume

    @classmethod
    def pdb_search(cls, protein: str):
        """Returns a PDBID for a given protein of interest"""
        search_result = pypdb.Query(protein)
        return search_result.search()

    @classmethod
    def get_mtz_file(cls, using_path_name: str):
        """Returns the contents of an MTZ file relevant to a search
        of a given protein in the PBDatabase. File data found is
        written to a file in the current working directory under
        the same name as the name passed to the 'using_path_name'
        :argument."""
        mtz_file = os.open(path="/Users/" + using_path_name, flags=os.O_WRONLY)
        protein_file_path = os.path.abspath(path=using_path_name)
        with os.open(path=protein_file_path, flags=os.O_RDWR):
            for thing in os.listdir(path=protein_file_path):
                os.write(__fd=mtz_file, __data=thing.encode())
        os.close(mtz_file)
        return mtz_file

    @classmethod
    def rotate_protein_around_x_axis(cls, protein_id: str, angle: float):
        """Rotates a protein around the x-axis by a given angle,
        determined by translation of the cursor."""
        pass

    @classmethod
    def rotate_protein_around_y_axis(cls, protein_id: str, angle: float):
        """Rotates a protein around the y-axis by a given angle,
                determined by translation of the cursor."""
        pass

    @classmethod
    def rotate_protein_around_z_axis(cls, protein_id: str, angle: float):
        """Rotates a protein around the z-axis by a given angle,
                determined by translation of the cursor."""
        pass

    @classmethod
    def get_protein_binding_kinetics(cls, protein_id: str):
        """Returns binding kinetics for the protein
        specified by the 'protein_id' argument."""
        pass

    @classmethod
    def get_protein_ligands(cls, protein_id: str):
        """Returns ligands of the protein
        specified by the 'protein_id' argument."""
        pass

    @classmethod
    def get_protein_domains(cls, protein_id: str):
        """Returns domains of the protein
        specified by the 'protein_id' argument."""
        pass

    @classmethod
    def get_protein_size(cls, protein_id: str):
        """Returns the size of the protein
        specified by the 'protein_id' argument."""
        pass

    @classmethod
    def get_protein_tertiary_structure(cls, protein_id: str):
        """Returns the tertiary structure of the protein
        specified by the 'protein_id' argument."""
        pass

    @classmethod
    def split_screen(cls):
        """Divides the drawing window of the screen in half,
        with one side of the drawing window for viewing, and
        the other side of the drawing window for editing."""
        pass

    @classmethod
    def toggle(cls):
        """Toggles which side of the drawing window that can be
        edited."""
        pass

    @classmethod
    def build_new_protein(cls):
        """This method prevents further editing of a drawing in the
         editing window."""
        pass

    @classmethod
    def add_bond(cls, position: tuple, atom_id: str):
        """This method creates a new bond originating from the element
        clicked on in the drawing window. The element, specific by the
        'element_id' argument, is added to the other side of the bond
        that is created."""
        pass

    """Below is a generative model that can 
    return a suggested tertiary structure for a protein, given 
    desired characteristics as an input string."""

    print("CPU count: " + str(os.cpu_count()))


class GAN(keras.Model):

    def __init__(self):
        super().__init__()

        # Define the layers of the GAN.
        self.encoder = keras.models.Sequential(name="Encoder")

        self.encoder_input_layer = keras.layers.InputLayer(
            batch_shape=CDQueryIdentifier.reshaped_one_hot_encoded_training_data.shape,
            name="Encoder_Input_Layer")

        self.encoder_dense_layer_one = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_One")
        self.encoder_leaky_relu_layer_one = keras.layers.LeakyReLU(alpha=0.1)

        self.encoder_dense_layer_two = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_Two")
        self.encoder_leaky_relu_layer_two = keras.layers.LeakyReLU(alpha=0.1)

        self.encoder_dense_layer_three = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_Three")
        self.encoder_leaky_relu_layer_three = keras.layers.LeakyReLU(alpha=0.1)

        self.encoder_dense_layer_four = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_Four")
        self.encoder_leaky_relu_layer_four = keras.layers.LeakyReLU(alpha=0.1)

        self.encoder_dense_layer_five = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_Five")
        self.encoder_leaky_relu_layer_five = keras.layers.LeakyReLU(alpha=0.1)

        """print("encoder_leaky_relu_layer_five shape: " + encoder_leaky_relu_layer_five.compute_output_shape(encoder_leaky_relu_layer_five.input))"""

        self.encoder_dense_layer_six = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_Six")
        self.encoder_leaky_relu_layer_six = keras.layers.LeakyReLU(alpha=0.1)

        self.decoder = keras.models.Sequential(name="Decoder")

        self.decoder_dense_layer_one = keras.layers.Dense(units=400, activation='relu', name="Decoder_Dense_Layer_One")
        self.decoder_leaky_relu_layer_one = keras.layers.LeakyReLU(alpha=0.1)

        self.decoder_dense_layer_two = keras.layers.Dense(units=20, activation='relu', name="Decoder_Dense_Layer_Two")
        self.decoder_leaky_relu_layer_two = keras.layers.LeakyReLU(alpha=0.1)

        self.decoder_dense_layer_three = keras.layers.Dense(units=20, activation='relu', name="Encoder_Dense_Layer_Three")
        self.decoder_leaky_relu_layer_three = keras.layers.LeakyReLU(alpha=0.1)

        self.decoder_dense_layer_four = keras.layers.Dense(units=20, activation='relu', name="Decoder_Dense_Layer_Four")
        self.decoder_leaky_relu_layer_four = keras.layers.LeakyReLU(alpha=0.1)

        self.decoder_dense_layer_five = keras.layers.Dense(units=20, activation='relu', name="Decoder_Dense_Layer_Five")
        self.decoder_leaky_relu_layer_five = keras.layers.LeakyReLU(alpha=0.1)

        self.decoder_dense_layer_six = keras.layers.Dense(units=20, activation='relu', name="Decoder_Dense_Layer_Six")
        self.decoder_leaky_relu_layer_six = keras.layers.LeakyReLU(alpha=0.1)

    def __call__(self, inputs):
        # Implement the forward pass of the GAN.
        y1 = self.encoder_dense_layer_one(inputs=self.encoder_input_layer.output)
        y2 = self.encoder_leaky_relu_layer_one(y1)
        y3 = self.encoder_dense_layer_two(y2)
        y4 = self.encoder_leaky_relu_layer_two(y3)
        y5 = self.encoder_dense_layer_three(y4)
        y6 = self.encoder_leaky_relu_layer_three(y5)
        y7 = self.encoder_dense_layer_four(y6)
        y8 = self.encoder_leaky_relu_layer_four(y7)
        y9 = self.encoder_dense_layer_five(y8)
        y10 = self.encoder_leaky_relu_layer_five(y9)
        y11 = self.encoder_dense_layer_six(y10)
        y12 = self.encoder_leaky_relu_layer_six(y11)
        y13 = self.decoder_dense_layer_one(y12)
        y14 = self.decoder_leaky_relu_layer_one(y13)
        y15 = self.decoder_dense_layer_two(y14)
        y16 = self.decoder_leaky_relu_layer_two(y15)
        y17 = self.decoder_dense_layer_three(y16)
        y18 = self.decoder_leaky_relu_layer_three(y17)
        y19 = self.decoder_dense_layer_four(y18)
        y20 = self.decoder_leaky_relu_layer_four(y19)
        y21 = self.decoder_dense_layer_five(y20)
        y22 = self.decoder_leaky_relu_layer_five(y21)
        y23 = self.decoder_dense_layer_six(y22)
        return self.decoder_leaky_relu_layer_six(y23)

    gan_loss = keras.losses.SparseCategoricalCrossentropy()

    gan_optimizer = keras.optimizers.SGD(learning_rate=0.001)

    gan_metrics = [keras.metrics.SparseCategoricalCrossentropy()]

    """Use a custom method to train the GAN model."""

    @staticmethod
    def fit_model(epochs: int):
        """
        :param epochs:
        :return the loss and metrics from training the model.:
        """
        x_train = CDQueryIdentifier.reshaped_one_hot_encoded_training_data
        if keras.utils.is_keras_tensor(x=x_train):
            x_train = keras.utils.get_source_inputs(tensor=x_train)
        else:
            print("I don't know what type x_train is!")

        model = GAN()
        model.compile(optimizer=GAN.gan_optimizer, loss=GAN.gan_loss, metrics=GAN.gan_metrics)
        print(str(x_train.shape))
        for _ in numpy.arange(start=0, stop=epochs, step=1):
            for _ in [GAN.layers]:
                for thing in x_train:
                    if thing is not None:
                        """x_train.take(thing, axis=0)"""
                    """return model.train_on_batch(x=x_train)"""
                    pass
            for thing in x_train:
                if thing is not None:
                    """x_train.take(thing, axis=0)"""
                """return model.train_on_batch(x_train)"""
                pass


"""Train the GAN on the processed training data using the custom training method."""

history = GAN.fit_model(epochs=CDQueryIdentifier.epochs)

"""Plot the loss and metrics from training to evaluate the model."""

"""pyplot.plot(CDQueryIdentifier.epochs, history["SparseCategoricalCrossentropy Loss"], label="SparseCategoricalCrossentropy Loss")

pyplot.show()"""
