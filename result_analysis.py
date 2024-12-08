from positional_encodings_transformer import forward, load_model_params, Tokenizer
import nltk
import re
import jax.numpy as jnp
import jax
import numpy as np
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon

# Get the corpus 
filename = 'Project\\bxk389\\sample_data.txt'
with open(filename, 'r') as file:
    text = file.read()
    text = text.strip()  # Remove leading/trailing spaces
    corpus = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
vocab_size = 40
t = Tokenizer(vocab_size=vocab_size)
t.train(corpus)
verbs = {'is' : 0, 'be' : 1, 'are' : 2, 'consume' : 3, 'consumed' : 4, 'eaten' : 5, 'baked' : 6, 'heated' : 7, 'bakes' : 8, 'heats' : 9}


probabilities = np.zeros(shape=(len(verbs), vocab_size))

# Calculate the probability of seeing some word directly after a verb
for i in range(len(corpus.split()) - 1):
    if corpus.split()[i] in verbs.keys(): # is a verb
        probabilities[verbs[corpus.split()[i]]][t.word_to_id[corpus.split()[i + 1]]] += 1

# Normalize to create a valid probability distribution
row_sums = probabilities.sum(axis=1)
probabilities = probabilities / row_sums[:, np.newaxis]




paramList = ['Project\\bxk389\\custom_pe_params.pkl', 'Project\\bxk389\\custom_pe_params_6.pkl', 'Project\\bxk389\\sinusoidal_pe_params.pkl', 'Project\\bxk389\\alibi_pe_params.pkl', 
          'Project\\bxk389\\rotary_pe_params.pkl', 'Project\\bxk389\\nope_pe_params.pkl']
modelNames = ['custom', 'custom_6', 'sinusoidal', 'alibi', 'rotary', 'nope']

# Code for Probability Analysis:
input_text = "\"carbohydrates are\""
print(f"Input text = {input_text}")
for i in range (len(paramList)):
    print(f"\nResults for {modelNames[i]} PE strategy:")
    params = load_model_params(model_params_path=paramList[i])
    input_ids = jnp.array([t.encode(input_text)])  # Wrapped in [] to make a batch of size 1, since methods expect a batch
    generated_tokens = input_ids

    logits = forward(generated_tokens, params)
    probs = np.array(logits[:, -1, :])

    print(f"Probability of next word = \'food\' = {probs[0][t.word_to_id['food']]}")

## Code for Distribution Analysis:
'''input_text = "\"consumed can be\""
print(f"Input text = {input_text}")
for i in range (len(paramList)):
    print(f"\nResults for {modelNames[i]} PE strategy:")
    params = load_model_params(model_params_path=paramList[i])

    input_ids = jnp.array([t.encode(input_text)])  # Wrapped in [] to make a batch of size 1, since methods expect a batch
    generated_tokens = input_ids

    logits = forward(generated_tokens, params)
    probs = np.array(logits[:, -1, :])

    # perform KL test between the logits and the dataset probabilities
    print(f"Jensen-Shannon Distance = {jensenshannon(probs[0], probabilities[verbs['be']])}")'''