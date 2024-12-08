##  How to train a model:
You will require both my code and my made-up dataset to train a model. Please download the files *sample_data.txt* and *positional_encoding_transformer.py*

In order to train any a model, one must first choose a combination of parameters. I have denoted the hyperparameters I used in parathesis. The options are:

- batch_size : positive int (32)
- sequence_length : positive int (4)
- embedding_dim : positive int (128)
- head_size : positive int (16)
- pe_strategy : str $\in$ ['custom', 'sinusoidal', 'alibi', 'rotary', 'nope']
- params_save_filename : str, 'whatever_filename.pkl'

These parameters can be specified starting on line 97 of *positional_encoding_transformer.py*

**Note**: please ensure that the value for *embedding_dim* evenly divies into the value for *head_size*.

Once you follow all of these steps, please run the python file and wait for your machine to finish training the model. The parameters will be saved in a *.pkl* file so you only need to run the model one time. 

## How to replicate results:

You will require my code to recreate the analysis. Please download the file *result_analysis.py*


By this point, you should have either trained the models you wanted or simply downloaded the *.pkl* files provided.

I performed two kinds of analysis on the trained models: distribution analysis and probability analysis. 

If you want to recreate my experiments for seeing the probabilty of a next word given an input sequence, please uncomment the labeled code block and run the python file.

If you want to recreate my experiments for comparing the model logits to the true distribution found in the dataset, please uncomment the labeled code and run the python file.

**Note**: since *result_analysis.py* imports functions from *positional_encoding_transformer.py*, it might be wise to comment out lines 549 and 550 in *positional_encoding_transformer.py*. This will ensure your computer isn't forced to train a new model each time you want to generate some results. 
