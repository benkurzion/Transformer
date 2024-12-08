from datetime import time

from datasets import load_dataset
import tiktoken
#import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
import jax
import numpy as np
import optax
from functools import partial
from random import randrange
import random
import math
from collections import defaultdict
import re, collections
import pickle
import re
from collections import Counter

# DONT TOUCH
class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.word_to_id[self.eos_token] = 0
        self.word_to_id[self.pad_token] = 1  # Assign ID to pad token
        self.id_to_word[0] = self.eos_token
        self.id_to_word[1] = self.pad_token  # Add pad token to the reverse mapping

    def clean_text(self, text):
        """Clean text by removing superfluous spaces and unwanted characters."""
        text = text.replace("'", "")  # Remove apostrophes
        text = text.strip()  # Remove leading/trailing spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    def train(self, corpus):
        corpus = self.clean_text(corpus)
        # Regex to capture words and punctuation separately
        words_and_punctuation = re.findall(r'\w+|[^\w\s]', corpus.lower())  # Match words and punctuation
        word_counts = Counter(words_and_punctuation)
        most_common_words = word_counts.most_common(self.vocab_size)

        for idx, (word, _) in enumerate(most_common_words, 2):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word



    def encode(self, text):
        text = text.replace("'", "")
        words_and_punctuation = re.findall(r'\b\w+\b|[^\w\s]', text.lower())  # \b\w+\b ensures we don't split <pad>
        tokens = []
        for word in words_and_punctuation:
            if word == "," or word == "'" or word == ";" or word == ":" or word == "\"":
                continue
            if word in {'.', '?', '!'}:  # EOS token trigger
                tokens.append(self.word_to_id[self.eos_token])
            else:
                token_id = self.word_to_id.get(word, self.word_to_id.get('<unk>', -1))
                tokens.append(token_id)

        tokens = list(map(int, tokens))

        return tokens

    def decode(self, tokens):
        decoded_text = []

        # Iterate over tokens and decode, ignoring <pad> tokens (ID 1)
        for token in tokens:
            if token == 1:  # Skip <pad> token
                continue

            word = self.id_to_word.get(token, '<unk>')  # Get the word corresponding to the token ID
            decoded_text.append(word)

        return ' '.join(decoded_text)
    
    




'''
Hyperparameters:
please ensure that embedding_dim evenly divies into head_size

'''
with open('Project\\bxk389\\sample_data.txt', "r") as f:
    corpus = f.read()
batch_size = 32
sequence_length = 4
embedding_dim = 128
head_size = 16
vocab_size = 40  # Fixed; property of tokenizer
pe_strategy = 'custom'  # ['custom', 'sinusoidal', 'alibi', 'rotary', 'nope']
params_save_filename = 'custom_pe_params.pkl'
t = Tokenizer(vocab_size=vocab_size)
t.train(corpus)


# Define the processing function
def process(line):
    ids = t.encode(line.strip())  # Remove leading/trailing whitespace
    out = {'ids': ids, 'len': len(ids)}
    return out

# Process each line and store results in a list
processed_data = []
for line in corpus.splitlines():
    processed_data.append(process(line))

# Convert processed data to a NumPy array
data_np = np.concatenate([p['ids'] for p in processed_data])

# Convert NumPy array to JAX array
data_jax = jnp.asarray(data_np)


# HELPER METHODS: DO NOT TOUCH
def softmax(z: jnp.ndarray):
    # Shift by max for numerical stability
    shifted_z = z - jnp.max(z, axis=-1, keepdims=True)

    numerator = jnp.exp(shifted_z)
    denominator = jnp.sum(numerator, axis=-1, keepdims=True) + 1e-10

    return numerator / denominator


'''
Relevant Dimensions:
query_dim = embedding_dim / head_size
Q,K,V = [batch_size, head_size, seq_len, query_dim]
W_q,k = [batch_size, embedding_size, embedding_size]
'''
def sinusoidal_position_encoding(embedding_matrix: np.ndarray) -> np.ndarray:
    '''Sinusoidal positional embeddings from the Vaswani paper'''
    batch_size, seq_len, embedding_dim = embedding_matrix.shape
    assert embedding_dim % 2 == 0  # must be even dimension
    for b in range(batch_size):
        for i in range(seq_len):
            positional_vector = np.zeros(shape=(embedding_dim))
            for m in range(0, len(positional_vector), 2):
                w = 10000 ** (2 * m / embedding_dim)
                positional_vector[m] = (math.sin(i / w))
                positional_vector[m + 1] = (math.cos(i / w))
            # Add the positional vector to the original embedding
            embedding_matrix[b, i] = embedding_matrix[b, i] + np.array(positional_vector)

    return embedding_matrix

def rotary_positional_encoding(Q: np.ndarray, K : np.ndarray):
    '''
    Rotational positional embeddings (RoPE)
    Applied to the query matrix Q and the key matrix K before the operation Q * transpose(K)
    '''

    batch_size, head_size, seq_len, embedding_dim = Q.shape
    assert embedding_dim % 2 == 0  # must be even dimension

    def get_theta(i: int):
        return 10000 ** (2 * (i - 1) / embedding_dim)

    def getRotationMatrix(m  : int):
        rotation_matrix = np.zeros(shape=(embedding_dim, embedding_dim))
        for i in range(0, len(rotation_matrix), 2):
            theta = get_theta((i / 2) + 1)
            rotation_matrix[i][i] = np.cos(m * theta)
            rotation_matrix[i][i + 1] = -1 * np.sin(m * theta)
            rotation_matrix[i + 1][i] = np.sin(m * theta)
            rotation_matrix[i + 1][i + 1] = np.cos(m * theta)
        return rotation_matrix
    
    S = np.zeros(shape=(batch_size, head_size, seq_len, seq_len)) # stores attention scores
    for b in range(batch_size):
        for h in range(head_size):
            for i in range (seq_len):
                for j in range (seq_len):
                    S[b][h][i][j] = jnp.matmul(np.transpose(Q[b][h][i]), jnp.matmul(getRotationMatrix(m=i - j), K[b][h][j]))
    return jnp.array(S)


def alibi_positional_encoding(S, num_heads : int):
    '''
    ALiBi Positional Encoding implementation
    Relative postional linear bias on raw attention scores
    '''
    S = np.array(S)
    batch_size, head_size, seq_len, _ = S.shape

    def getM (head_index : int, num_heads : int):
        initial_m = 1/(2 ** (8/num_heads))
        m = initial_m
        for _ in range (head_index):
            m = m * initial_m
        return m
    
    distance_matrix = np.zeros(shape=(seq_len, seq_len))
    for i in range (seq_len):
        for j in range (seq_len):
            if i > j :
                distance_matrix[i][j] = j - i
    for b in range(batch_size):
        for h in range(head_size):
            S[b][h] = S[b][h] + distance_matrix * getM(head_index=h, num_heads=num_heads)
    return jnp.array(S)


def custom_pe(E : np.ndarray):
    '''

    MY EXTENSION:

    '''
    batch_size, sequence_length, embedding_dim = E.shape
    def getRotationMatrix(position : int):
        '''Returns a rotation matrix that rotates a vector by position* 2pi/36 radians counterclockwise'''
        rotation_matrix = np.zeros(shape=(embedding_dim, embedding_dim))
        for i in range(0, len(rotation_matrix), 2):
            theta = 2 * math.pi / 36
            rotation_matrix[i][i] = np.cos(position * theta)
            rotation_matrix[i][i + 1] = -1 * np.sin(position * theta)
            rotation_matrix[i + 1][i] = np.sin(position * theta)
            rotation_matrix[i + 1][i + 1] = np.cos(position * theta)
        return rotation_matrix
    for b in range(batch_size):
        for s in range(sequence_length):
            scale = (math.floor(s/36) + 1) * 0.1 * np.linalg.norm(E[b][s])
            E[b][s] = scale * np.matmul(getRotationMatrix(position=s), E[b][s])
    return E

@partial(jax.jit, static_argnames=('batch_size', 'seq_len'))
def get_batch(random_key, data, batch_size, seq_len):
  idxs = jax.random.randint(
      random_key, shape=(batch_size, 1), minval=0, maxval=len(data) - seq_len - 1
  )
  return jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(data, idxs, (seq_len + 1,))


# Should take a *batch* of token ids
# input_ids is a 2D array with dimensions (batch_size x sequence_length) 
def forward(input_ids : jnp.ndarray, params : dict):
    batch_size, sequence_length = input_ids.shape
    query_size = int(embedding_dim / head_size)

    # Translate the tokenized sequence to the corresponding embeddings
    embedded_tokens = np.zeros(shape=(batch_size, sequence_length, embedding_dim))
    for b in range(batch_size):
        for s in range(sequence_length):
            embedded_tokens[b][s] = params['embeddings'][input_ids[b][s]]

    # Sinusoidal positional embeddings
    if pe_strategy == 'sinusoidal':
        embedded_tokens = sinusoidal_position_encoding(embedding_matrix=embedded_tokens)
    elif pe_strategy == 'custom':
        embedded_tokens = custom_pe(E=embedded_tokens)
    E = jnp.array(embedded_tokens)

    # Calculate the query, key and value matrices and reshape them to accommodate multiple heads
    Q = jnp.matmul(E, params['W_q'][:batch_size])
    Q = Q.reshape(batch_size, head_size, sequence_length, query_size)
    K = jnp.matmul(E, params['W_k'][:batch_size])
    K = K.reshape(batch_size, head_size, sequence_length, query_size)
    V = jnp.matmul(E,params['W_v'][:batch_size])
    V = V.reshape(batch_size, head_size, sequence_length, query_size)

    # Calculate attention score
    # Mask
    mask = jnp.zeros((sequence_length, sequence_length))
    indices = jnp.triu_indices(sequence_length, k=1)
    mask = mask.at[indices].set(-jnp.inf)

    query_size = int(embedding_dim / head_size)
    scale = 1 / jnp.sqrt(query_size)
    if pe_strategy == 'rotary':
        S = scale * rotary_positional_encoding(Q=Q, K=K) + mask
    elif pe_strategy == 'alibi':
        S = scale * jnp.matmul(Q, jnp.transpose(K, axes=(0,1,3,2))) + mask
        S = alibi_positional_encoding(S=S, num_heads=head_size)
    else:
        S = scale * jnp.matmul(Q, jnp.transpose(K, axes=(0,1,3,2))) + mask
    A = softmax(S)
    O = jnp.matmul(A, V)
    O = jnp.transpose(O, axes=(0,2,1,3))
    # This is the same thing as C_o in our original implementation 
    O = O.reshape(batch_size, sequence_length, head_size * query_size)

    P_o = jnp.matmul(O, params['W_o'])

    H = P_o + E

    logits = jnp.matmul(H, params['W_out'])

    # Add all of the matrices to params
    params['Q'] = Q
    params['K'] = K
    params['V'] = V
    params['S'] = S
    params['A'] = A
    params['O'] = O
    params['P_o'] = P_o
    params['H'] = H
    params['E'] = E

    return softmax(logits)  # (B, T, vs)


# input_ids and target are (B, T)
def backward(logits, target, params):
    query_size = int(embedding_dim / head_size)

    # Encode target array as a one hot encoding matrix
    # Since we use the encoded value as the 1 index in the one hot encoding, we already know the target vectors
    # Given a token x, then the corresponding OHE = [0, 0, ..., x, 0, 0...] where the xth index is 1
    temp = np.zeros(shape=logits.shape)
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            temp[b][s] = np.zeros(shape=(vocab_size))
            temp[b][s][target[b][s]] = 1
    target = jnp.array(temp)

    loss_wrt_logits = (logits - target)

    loss_wrt_W_out = jnp.matmul(jnp.transpose(params['H'], axes=(0, 2, 1)), loss_wrt_logits)

    loss_wrt_H = jnp.matmul(loss_wrt_logits, jnp.transpose(params['W_out'], axes=(0,2,1)))

    loss_wrt_P_o = loss_wrt_H

    loss_wrt_E = loss_wrt_H

    # Since we are moving back through the forward pass, we unpack O so it is indexed by head_size
    loss_wrt_O = jnp.matmul(loss_wrt_P_o, jnp.transpose(params['W_o'], axes=(0,2,1)))
    loss_wrt_O = loss_wrt_O.reshape(batch_size, head_size, sequence_length, query_size)

    loss_wrt_W_o = jnp.matmul(jnp.transpose(params['O'], axes=(0,2,1)), loss_wrt_P_o)

    loss_wrt_A = jnp.matmul(loss_wrt_O, jnp.transpose(params['V'], axes=(0, 1, 3, 2)))

    loss_wrt_V = jnp.matmul(jnp.transpose(params['A'], axes=(0, 1, 3, 2)), loss_wrt_O)
    loss_wrt_V = loss_wrt_V.reshape(batch_size, sequence_length, head_size * query_size)

    loss_wrt_W_v = jnp.matmul(jnp.transpose(params['E'], axes=(0, 2, 1)), loss_wrt_V)
    
    loss_wrt_E += jnp.matmul(loss_wrt_V, jnp.transpose(params['W_v'], axes=(0, 2, 1)))

    loss_wrt_S = np.zeros(shape=loss_wrt_A.shape)

    for b in range(loss_wrt_S.shape[0]):  # batch dimension
        for h in range(loss_wrt_S.shape[1]): # head dimension
            for s1 in range(loss_wrt_S.shape[2]):  #sequence dimension
                for s2 in range(s1):  # second seqence dimension with mask
                    if s1 == s2:
                        loss_wrt_S[b][h][s1][s2] = loss_wrt_A[b][h][s1][s2] * (1 - loss_wrt_A[b][h][s1][s2])
                    else:
                        loss_wrt_S[b][h][s1][s2] = -1 * loss_wrt_A[b][h][s1][s2] * loss_wrt_A[b][h][s2][s1]

    loss_wrt_S = jnp.array(loss_wrt_S)

    loss_wrt_Q = (1 / jnp.sqrt(query_size)) * jnp.matmul(loss_wrt_S, params['K'])

    loss_wrt_K = (1 / jnp.sqrt(query_size)) * jnp.matmul(jnp.transpose(params['Q'], axes=(0, 1, 3, 2)), loss_wrt_S)

    # Squashing to 3D
    loss_wrt_Q = loss_wrt_Q.reshape(batch_size, sequence_length, head_size * query_size)
    loss_wrt_K = loss_wrt_K.reshape(batch_size, sequence_length, head_size * query_size)

    loss_wrt_W_q = jnp.matmul(jnp.transpose(params['E'], axes=(0, 2, 1)), loss_wrt_Q)

    loss_wrt_W_k = jnp.matmul(jnp.transpose(params['E'], axes=(0, 2, 1)), loss_wrt_K)

    loss_wrt_E += jnp.matmul(loss_wrt_K, jnp.transpose(params['W_k'], axes=(0, 2, 1)))

    loss_wrt_E += jnp.matmul(loss_wrt_Q, jnp.transpose(params['W_q'], axes=(0, 2, 1)))

    grads = {}

    grads['E'] = loss_wrt_E
    grads['W_k'] = loss_wrt_W_k
    grads['W_q']=  loss_wrt_W_q
    grads['W_v']= loss_wrt_W_v
    '''
    grads['H'] =loss_wrt_H
    grads['O'] = loss_wrt_O
    grads['A'] = loss_wrt_A
    grads['S'] = loss_wrt_S
    grads['V'] = loss_wrt_V 
    '''
    grads['W_out'] = loss_wrt_W_out
    grads['W_o'] = loss_wrt_W_o

    return grads

def update_embeddings(params, batch, grads):
    B, T, C = params['E'].shape

    final_loss_wrt_e = grads['E']


    embedding_matrix = params['embeddings']
    # Creating a matrix which stores the gradients for the input embedding
    # Has same shape as the embedding lookup matrix(size = (vocab_size, vocab_size))
    embedding_grads = np.zeros_like(embedding_matrix)

    for b in range(B):
        for s in range(T):
            token_id = batch[b, s]
            embedding_grads[token_id] = embedding_grads[token_id] +final_loss_wrt_e[b, s]
    return embedding_grads


# logits is (B, T, vs)
# target is (B, T)
# DONT TOUCH
def calculate_loss(logits, target):
    loss = 0
    # Convert the tokens in target to their OHE vectors
    # target will become (B, T, vs)
    # Since we use the encoded value as the 1 index in the one hot encoding, we already know the target vectors
    # Given a token x, then the corresponding OHE = [0, 0, ..., x, 0, 0...] where the xth index is 1
    temp = np.zeros(shape=logits.shape)
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            temp[b][s] = np.zeros(shape=(vocab_size))
            temp[b][s][target[b][s]] = 1
    target = jnp.array(temp)
    batch_size = len(target)
    # calculate cross entropy loss over all batch
    loss = (-jnp.sum(target * jnp.log(logits)))/batch_size

    return loss

def train(params, iterations, num_epochs, batch_size):
    optimizer = optax.adam(0.0003)
    embeddings = params['embeddings']
    param_keys = ['W_q', 'W_k', 'W_v', 'W_o', 'W_out', 'embeddings']
    trainable_params = {}
    for k in param_keys:
        trainable_params[k] = params[k]

    trainable_grads = {}
    opt_state = optimizer.init(trainable_params)

    key = jax.random.PRNGKey(1337)  # For sampling batches

    for epoch in range(num_epochs):
        pbar = tqdm(range(iterations))
        for i in pbar:
            key, subkey = jax.random.split(key)
            batch = get_batch(subkey, data_jax, batch_size, sequence_length)

            input_ids = batch[:, :-1]
            target = batch[:, 1:]

            # Feel free to deviate from these signatures
            logits = forward(input_ids, trainable_params)
            loss = calculate_loss(logits, target)
            grads = backward(logits, target, trainable_params)
            embedding_grads = update_embeddings(trainable_params, batch, grads)
            grads['embeddings'] = embedding_grads
            # keeps the gradients for the necessary parameters
            for k in param_keys:
                trainable_grads[k] = grads[k]
            # only keep the parameters that will be updated
            trainable_params = {k: v for k, v in trainable_params.items() if k in param_keys}

            #trainable_grads['embeddings'] = embedding_grads
            updates, opt_state = optimizer.update(trainable_grads, opt_state)
            trainable_params = optax.apply_updates(trainable_params, updates)

            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")
    
    with open(params_save_filename, 'wb') as f:
        pickle.dump(trainable_params, f)

    return trainable_params 

def load_model_params(model_params_path):
    with open(model_params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params
    
def initialize_params(embedding_dim : int, head_size : int, vocab_size : int) -> dict:
    # We assume that each token is unique
    # embeddings[token_id, :] = a vector of dimension (embedding_dim x 1) representing the embedding for token_id
    embeddings = np.zeros(shape=(vocab_size, embedding_dim))
    
    # The initial embeddings are sampled from N(0,std)
    std = jnp.sqrt(1 / embedding_dim)
    for i in range (len(embeddings)):
        embeddings[i] = np.random.normal(loc=0, scale=std, size=embedding_dim)

    embeddings = jnp.array(embeddings)



    # Now we initialize W_q, W_k, W_v, W_out, and W_o for every head into one big matrix
    # FOLLOW THIS FOR STEPS
    # https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
    query_size = int(embedding_dim / head_size)
    attn_std = jnp.sqrt(1.0 / (embedding_dim + query_size))  # or jnp.sqrt(1/embedding_dim)
    output_std = jnp.sqrt(1.0 / (embedding_dim + vocab_size))
    wo_std = jnp.sqrt(1.0 / (embedding_dim + embedding_dim))
    

    W_q = np.random.normal(0, attn_std, size=(batch_size, embedding_dim, embedding_dim))

    W_k = np.random.normal(0, attn_std, size=(batch_size, embedding_dim, embedding_dim))

    W_v = np.random.normal(0, attn_std, size=(batch_size, embedding_dim, embedding_dim))

    W_o = np.random.normal(0, wo_std, size=(batch_size, embedding_dim, embedding_dim))

    W_out = np.random.normal(0, output_std, size=(batch_size, embedding_dim, vocab_size))

    params = {}
    params['embeddings'] = embeddings
    params['W_q'] = W_q
    params['W_k'] = W_k
    params['W_v'] = W_v
    params['W_out'] = W_out
    params['W_o'] = W_o
    return params

params = initialize_params(embedding_dim=embedding_dim, head_size=head_size, vocab_size=vocab_size)
params = train(params, iterations=50, num_epochs=10, batch_size=batch_size)