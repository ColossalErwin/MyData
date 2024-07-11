# Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings as warn

# Hosting stuff (to ignore verification error)
import urllib.request
from urllib.request import urlopen
import ssl
import json


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

class NeuralNet:
    def __init__(self, dataFile, header = True):
        self.raw_input = pd.DataFrame(pd.read_csv(dataFile, header = None))
        
        # Loaded Successfully
        print("Data Loaded Successfully")
        print("Spambase Data Set has {} data points with {} variables each.".format(*self.raw_input.shape))
        print(self.raw_input)

    # Preprocessing Stage
    def preprocess(self):
        print("\nPre-Processing the Data:\n")
        self.processed_data = self.raw_input

        # Check for null values in the dataframe
        print("Null entries found?:", ("No\n" if self.processed_data.isnull().sum().sum() == 0 else "Yes\n"))

        # Check for duplicate values in the dataframe
        print("Duplicate entries found?:", ("No\n" if self.processed_data.duplicated().sum() == 0 else "Yes\n"))

        # Remove the duplicate columns
        print("Removing all the duplicate entries\n")
        self.processed_data = self.processed_data.drop_duplicates()

        # Check for duplicate values in the dataframe again
        print("Duplicate entries found?:", ("No\n" if self.processed_data.duplicated().sum() == 0 else "Yes\n"))
        
        # Check if there is any categorical values
        print("Check for categorical values:")
        print(self.processed_data.dtypes)

        # The dataset contains no column headers, so we add them manually for easy analysis
        print("\nFor easy analysis, we rename the column headers\n")
        new_columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
                        'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people',
                        'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
                        'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
                        'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
                        'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 
                        'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
                        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'Class']
        self.processed_data.columns = new_columns
        print(self.processed_data)
        print(self.processed_data.describe())

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        print("\nTraining the model:\n")
        # Initializing important variables
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Declaring lists to store required results
        train_accuracy_list = []
        test_accuracy_list = []
        train_mse_list = []
        test_mse_list = []
        activation_list = []
        learning_rate_list = []
        iterations_list = []
        hidden_layers_list = []

        # Plot related variables declared
        plt_count = 1
        main_plt = plt.figure(edgecolor = 'black', linewidth = 4)
        main_plt.set_figheight(50)
        main_plt.set_figwidth(40)

        # Scaling the data set
        std_scaler = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)

        # Hyperparameters for evaluation and analysis
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        for function in activations:
          for rate in learning_rate:
            for iterations in max_iterations:
              for size in num_hidden_layers:
                # Store training parameters
                activation_list.append(function)
                learning_rate_list.append(rate)
                iterations_list.append(iterations)
                hidden_layers_list.append(size)
                tmp_train_accuracy = []
                tmp_test_accuracy = []
                tmp_iterations = []

                # Initialize the model
                model = MLPClassifier(activation = function, learning_rate_init = rate, max_iter = iterations, hidden_layer_sizes = size)

                # For history of model while training
                for i in range (1, iterations + 1, 1):
                  tmp_iterations.append(i)
                  
                  # Fit for one iteration
                  model.partial_fit(X_train,y_train, classes = np.unique(y_train))

                  # Test on train data
                  y_pred = model.predict(X_train)
                  y_pred = (y_pred > 0.5).flatten().astype(int) 
                  accuracy = accuracy_score(y_train, y_pred)
                  mse = mean_squared_error(y_train, y_pred)

                  # Store test results
                  tmp_train_accuracy.append(accuracy)

                  # Only store final result in the result table
                  if(i == iterations):
                    train_accuracy_list.append(accuracy)
                    train_mse_list.append(mse)

                  # Test on test data
                  y_pred = model.predict(X_test)
                  y_pred = (y_pred > 0.5).flatten().astype(int) 
                  accuracy = accuracy_score(y_test, y_pred)
                  mse = mean_squared_error(y_test, y_pred)

                  # Store test results
                  tmp_test_accuracy.append(accuracy)

                  # Only store final result in the result table
                  if(i == iterations):
                    test_accuracy_list.append(accuracy)
                    test_mse_list.append(mse) 

                # Plot for each model
                print(plt_count, "Plotting results with parameters:",'Function =', function, 'Epochs = ' + str(iterations) + ' Lr = ' + str(rate) + ' Hidden Layers = ' + str(size))
                figure = main_plt.add_subplot(8,3,plt_count)
                figure.plot(tmp_train_accuracy, label = 'Training')
                figure.plot(tmp_test_accuracy, label = 'Testing')
                figure.set_xlabel('epochs')
                figure.set_ylabel('test and train accuracy')
                figure.set_title('Function = ' + function + ' Epochs = ' + str(iterations) + ' Lr = ' + str(rate) + ' Hidden Layers = ' + str(size))
                figure.legend(loc = "lower right")
                plt_count += 1

        result_table = pd.DataFrame({'Activation':activation_list,
                              'Learning_Rate':learning_rate_list,
                              'Max_Iterations':iterations_list,
                              'Hidden_Layers':hidden_layers_list,
                              'Accuracy_train':train_accuracy_list,
                              'MSE_train':train_mse_list,
                              'Accuracy_test':test_accuracy_list,
                              'MSE_test':test_mse_list})

        result_table.index = result_table.index + 1

        result_table.to_csv('results.csv')
        print("\nPrinting the required output table:\n")
        print(result_table)
        

if __name__ == "__main__":
    #ignore verification error
    ssl._create_default_https_context = ssl._create_unverified_context
    neural_network = NeuralNet("https://github.com/ColossalErwin/MyData/blob/main/spambase.data?raw=true") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()