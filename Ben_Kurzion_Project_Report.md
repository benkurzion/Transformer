# Ben Kurzion - Investigating Transformer Positional Encodings
## Survey

### Sinusoidal PE [1]

The "Attention Is All You Need" paper discusses a great deal of novel architecture (the transformer), but one of the many elements they go over is their absolute positional encoding strategy. This paper uses sinusoidal positional encodings which is expressed as a positional vector that is added to the initial word embeddings of dimension $d$. To calculate the position vector to add to the embedding, they use sine and cosine functions. For each pair of embedding dimensions  $[(0,1), (2,3), ..., (d - 1, d)]$, a pair of corresponding values are calculated to indicate position. Formally, given two dimensions at indices $i$ and $i+1$ which represent pair $m$ in the embedding vector for some token at sequence position $p$,

$w = 10,000^{\text{2 * m / d}}$

so $posvec_i = sin(p/w)$ and  $posvec_{\text{i + 1}} = cos(p/w)$

Finally, the embedding vector is updated:

$embedvec = embedvec + posvec$

To observe the behavior of positional encoding strategy, I plotted the first 8 positions for a 2D vector $[1,1]$

<img src="https://github.com/benkurzion/Transformer/blob/main/sinusoidal_pe_image.png" alt="drawing" width="400"/>

As we can see, there is almost no discernable pattern between the different positions. The vector is being stretched, shrunk, and rotated in a very unpredictable way. This behavior would be hard for a machine learning model to generalize as there is no clear pattern. 

As for results, this paper makes one comparison between their sinusoidal positional encoding and a learned positional encoding strategy. They found similar results for both strategies. Seeing as this paper was not primarily meant to discuss positional encoding, this sparse testing is reasonable. 

### Rotary PE [2]

This paper focused on a mixture of both absolute and relative positional encoding. The intuition behind this paper is that in a standard transformer model, the attention score for two words at positions $i$ and $j$ respectively is usually only a function of the embeddings $x_i, x_j$:
$score = q_i ^ T * k_j$ where $q_i = W_q * x_i$ and $k_j = W_k * x_j$.
This paper changed this calculation so that the attention score incorporated the relative distance $(i-j)$. Rather than calculate the query and key vectors the normal way, the paper multiplied by a rotational matrix:

$q_i = R_i ^d * W_q * x_i$ and $k_j = R_j ^d * W_k * x_j$ 

where $R_t ^d =$

$$  \begin{bmatrix}
    cos(t*\theta_1) & -sin(t*\theta_1) & 0 & 0 & ... & 0 \\
    sin(t*\theta_1) & cos(t*\theta_1)& 0 & 0 & ... & 0 \\
    0 & 0 & cos(t*\theta_2) & -sin(t*\theta_2) & ... & 0 \\
    0 & 0 & sin(t*\theta_2) & cos(t*\theta_2) & ... & 0 \\ 
    ... & ... & ... &... & ... & ... & \\ 
    0 & ... & ... & ... & cos(t*\theta_{\frac{d}{2}}) & -sin(t*\theta_{\frac{d}{2}}) \\ 
    0 & ... & ... & ... & sin(t*\theta_{\frac{d}{2}}) & cos(t*\theta_{\frac{d}{2}})
    \end{bmatrix}$$

and $\theta_i = 10,000^{\frac{-2(i-1)}{d}}$

and $d = embedding \\_ dim$ 

The paper argued that simply multiplying the query and key vectors by $R_t ^d$ would encode absolute positions; vectors that have a larger position in the input sequence are rotated more counterclockwise than vectors that have a smaller position in the input sequence. This logic was unclear to me as the paper is rotating query and key vectors, not embeddings. And while the difference between the query/key and corresponding embedding vector is a single linear transformation, I am not sure that a token's absolute position can be expressed properly in just a query or key vector. 

Next, the paper showed that the vector rotation and attention score calculation can be rewritten to express the relative position $(i-j)$:

$attention score = (R_i ^d * W_q * x_i)^T (R_j ^d * W_k * x_j)$ 

if $(R_i ^d)^T * (R_j ^d)= R_{\text{i-j}}^d$, then 

$= x^T * W_q * R_{\text{i-j}}^d * W_k * x_j$ which I believe is wrong. The correct outcome should be
$= x_i ^T * W_q ^T * R_{\text{i-j}}^d * W_k * x_j$

Using their incorrect derivation, the attention score between two tokens at positions $i$ and $j$ respectively (with random embeddings and weight matrices) is not inversely proportional to position.

| Distance  | Attention Score |
| ------------- | ------------- |
| 1  | -1.503  |
| 5  | 2.943  |
| 10  | 0.302  |
| 20  | -1.874  |

Even with my mathematical correction, the attention scores do not obey any kind of decay. I think the fundamental issue is in the paper's assertion that $(R_i ^d)^T * (R_j ^d)= R_{\text{i-j}}^d$. This is probably incorrect. Just rotating two vectors does not necessarily encode their relative position to one another.

As for results, the paper compares their method against two other positional encoding strategies, one of which is the sinusoidal encodings. The paper shows that on some NLP tasks like translation, their positional encoding method outperforms the others. 


### ALiBi PE [3]
This paper focused on an exclusively relative positional encoding scheme. This method adds positional information by biasing the attention scores calculated at each head in the model. 
When the model calculates the pairwise attention score between some query vector $q_i$ and some key vector $k_j$ which correspond to two input tokens at positions $i$ and $j$ respectively, the model will add some bias to the attention score to indicate the relative position $(i-j)$. Formally, the paper defines this bias as a matrix addition between the attention scores and a scaled bias matrix:

$M_{\text{attention}} = $

$$  \begin{bmatrix}
    q_1 * k_1 & - & - \\
    q_2 * k_1 & q_2 * k_2 & - \\
    q_3 * k_1 & q_3 * k_2 & q_3 * k_3 \\
    \end{bmatrix}$$

$M_{\text{bias}} = $

$$  \begin{bmatrix}
    0 & - & - \\
    -1 & 0 & - \\
    -2 & -1 & 0 \\
    \end{bmatrix}$$

And given $n$ heads, 

$$m_0 = 2^{-\frac{8}{n}} $$ --> the scale for the first head

$m_1 = m_0 * 2^{-\frac{8}{n}} $

$m_2 = m_1 * 2^{-\frac{8}{n}} $

...

$m_n = 2^{-8} $

Then the final transformation of the attention scores for some head $h$ is as follows 

$M_{\text{attention}} = M_{\text{attention}} + M_{\text{bias}} * m_h $


After the bias is applied to the attention scores, the model computes a softmax as usual. The paper does not specify whether or not they scale the attention score by $${\frac{1}{dim_q}}$$. For my implementation, I assumed that they kept this scaling in their implementation. 

ALiBi was primarily concerned with the generalization power a transformer using a positional encoding strategy has: how does perplexity behave as the model generates strings longer than any of the strings it saw during training? This paper evaluted ALiBi trained with $L$ length sequences versus the sinusoidal encodings trained with $2L$ length sequences. The paper showed that for a variety of datasets, ALiBi is able to maintain a better perplexity while being trained on a fraction of sequence length in comparison to sinusoidal positional encodings. And when trained on the same length sequences as sinusoidal, rotary, and T5, the ALiBi method achieved the superior perplexity.

### NoPE [4]

On the surface, this paper implemented the simplest positional encoding strategy: no positional encoding. The paper argues that a decoder-only transformer can learn both relative and absolute positional encodings. However, this is not for any decoder-only transformer; the paper carefully implemented its attention blocks to ensure the model would pick up on the token positions. The paper describes two theorems:

Theorem 1: Given an input sequence of length $T$, there exist a set of weight matrices $W_q,W_k,W_v,W_o,W_1,W_2$ that enable the first self-attention block and subsequent feed forward layer to recover the absolute positions of the input sequence. This information will be stored in the hidden state $H^1$

Theorem 2: Given that the first hidden state $H^1$ contains absolute positional information for each input token, then the subsequent attention blocks and feed forward layers in the decoder can recover the relative positional encoding between tokens such that the attention score between some query and key vectors $q_i$ and $k_j$ will be defined as

$f_{\text{cnt}}(q_i, k_j) + f_{\text{rel}}(j - i)$

where $f_{\text{cnt}}(q_i, k_j)$ is a function of the content of the query and key and $f_{\text{rel}}(j - i)$ is a function of the relative distance between the query and key. 

For theorem 1, the paper specifies how they define the weight matrices. For example, the weight matrix $W_k$ is defined as 

$$  \begin{bmatrix}
    1 & 0 & ... & 0 \\
    ... & 0 & ... & 0 \\
    ... & ... & ... & ... \\
    1 & 0 & ... & 0 \\
    \end{bmatrix}$$

The other weight matrices are equally sparse and specific. The final hidden state encodes position $t$ in the output vector 


$$  \begin{bmatrix}
    0 \\
    0 \\
    1/t \\
    ... \\
    0
    \end{bmatrix}$$

The initializations for the weight matrices in the second theorem are equally puzzling with very sparce rows, a lot of zeros, and highly specific. In their results, the paper continues to puzzle as they refuse to compare their model against other models using perplexity as the measure. Instead, they evaluate the models on their ability to generalize (long validation sequences) on basic reasoning tasks and math. 

I am not sure why this paper is so adamant in not using perplexity, but it is definetly fishy behavior. Furthermore, the way they define their model seems way too specific and I suspect will not generalize well. 
    
### Compare and Contrast

While these papers all try to solve positional encoding, they approach the problem from very different angles. Sinusoidal tried absolute encoding, Rotary tried a mix of absolute and relative, ALiBi tried strictly relative, and NoPE neglected both. The only real way to compare these papers is in their complexity. 

The sinusoidal and ALiBi are both simple to implement and interpret. Their results and desired outcome are unambiguous. 

However, the NoPE and rotary stretegies are both overly complicated. The rotary method is so complex it seems that even the authors got it wrong at points. The NoPE method, while simple sounding, is quite complex and likely over-constrained with all the specifications they demand. 

## Methods

For comparison purposes, I implemented the positional encodings as described in papers [1] - [3]. Namely, I implemented sinusoidal, rotary (exactly as the paper specified), and linear biases positional encoding strategies. However, in Section *B1* of the NoPE paper, 
the authors detail the model they used for their experiments. While they did their testing on a decoder-only transformer, they used a full decoder; an attention block followed by a feed forward MLP. Without this MLP layer, their theorems 
1 and 2 would not be applicable. So the approximate the performance of the NoPE paper, I evaluated our model using no positional encoding strategy and trained in the same way as I did with the other positional encoding strategies.

## Research

For my research extension, I formulated a new positional encoding strategy. As discussed, the sinusoidal positional encodings are not very conducive torwards learning as they are far too temperamental in the way they stretch and rotate.
However, I wanted to investigate how a more predictable absolute encoding strategy strategy would perform. To do this, I created my custom positional encoding which, unlike the sinusoidal strategy which adds a positional vector to the initial 
word embeddings, I directly rotate and stretch the embedding. I calculate the embedding transformation as follows:

Given some input token at absolute position $t$ in the input sequence, I define $\theta$ as fixed value:

$\theta=2\pi/36$

and a corresponding rotation matrix $R_t$ of size [embedding\_dim, embedding\_dim] as

$$  \begin{bmatrix}
    cos(t*\theta) & -sin(t*\theta) & 0 & 0 & ... & 0 \\
    sin(t*\theta) & cos(t*\theta)& 0 & 0 & ... & 0 \\
    0 & 0 & cos(t*\theta) & -sin(t*\theta) & ... & 0 \\
    0 & 0 & sin(t*\theta) & cos(t*\theta) & ... & 0 \\ 
    ... & ... & ... &... & ... & ... & \\ 
    0 & ... & ... & ... & cos(t*\theta) & -sin(t*\theta) \\ 
    0 & ... & ... & ... & sin(t*\theta) & cos(t*\theta)
    \end{bmatrix}$$

By multiplying the embedding vector by this rotation matrix, the embedding vector is rotated by a fixed amount counterclockwise around the origin. However, position 1 and position 36 (and all subsequent multiples of 36) will  be rotated the same amount which creates ambiguity. To truly differentiate, the embedding vectors are multiplied, or stretched, by a scalar derived as follows:

$scale= (floor(t/36) + 1) * 0.1 * magnitude(vec_{\text{embed,t}})$

So the final transformation can be formulated as:
$vec_{\text{embed,t}} = scale * R_t*vec_{\text{embed,t}}$

Once the transformation is applied to each embedding vector in the input sequence, the transformer performs its forward and backward passes as usual with no modification.


## Results, Analysis and Discussion

I performed 3 experiments to better understand the effect positional encodings have on a transformer. I will discuss each experiment one by one and go over the insights each offers. 

### Experiment 1

I tested the different positional encoding strategies in their capacity to understand symmetric relationships. For example, "A does B" is not the same thing as "B does A". In my dataset, I had a number of 3 word sentences of the construction [Subject Verb Adj]. Ideally, the positional encoding is robust enough to inform the model that [Subject Verb Adj] != [Adj Verb Subject]. To test this, I fed the model the input "food is" which is a stem of the full sentence "food is good" in the dataset. I compared the model's normalized next-word-logit with the dataset's next word distribution for the verb "is". The two distributions were compared using the Jensen-Shannon Distance metric. Lower scores indicate that the model produced a next-word distribution similar to that of the dataset.


| Sinusoidal | ALiBi | Rotary | NoPE | Custom |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.7223809222775 | 0.7044150776111189  | 0.7139650749814189 | 0.7168947197128057 | 0.6785690650771237 | 

In simple constructions like [Subject Verb Adj], we expect absolute encodings to perform better than relative. Absolute positional encodings consider a distinct ordering and therefore 123 != 321. The model is less likely to predict a word that usually comes before "food" because it has learned a certain word ordering that has "food is" first. However, relative encodings would not be able to differentiate 123 and 321 because words 1 and 3 retain the same relative distance to one another and to 2. 

These suspicions are consistent with the results. The sentence “food is good” appears in the training set. So when testing the divergence between distributions with the models and the dataset distribution, we saw the best performance from my custom PE which is strictly absolute. The method I used to differentiate words at different positions was clearly effective: the model was less likely to generate words out of order given the input sequence. 
The reason sinusoidal, another absolute positional encoding, didn’t follow this trend is because it is too unpredictable in the way it changes vectors. While embeddings at different positions are changed uniquely, the relationship between a vector at position 1 versus position 100 is not learnable.

As for the relative positional encoding strategies, rotary performed roughly the same as ALiBi; a strictly relative PE. This was puzzling as the math in the rotary paper seemed to suggest that rotary performance would be more in line with absolute encodings rather than relative (as the relative encoding math was wrong).

Finally, since NoPE isnt initialized in the way the paper recommends, the model is essentially a bag of words for which 123 = 321. It came as no surprise to see uncompetitive results from the NoPE model.

### Experiment 2

Once again, I wanted to test the position encodings ability to understand symmetric relationships, but this time, I reversed the order. In Experiment 1, I wanted to confirm that the better models are more likely to generate something correct. In Experiment 2, I wanted to see how the models perform when they are given something wrong. Are the better models going to be sensitive to incorrect word ordering in the prompt? I fed the model the prompt "good is", a sequence that does not appear in the dataset. 

| Sinusoidal | ALiBi | Rotary | NoPE | Custom | Custom_6
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.7194450237613623 | 0.6984908548248002  | 0.7007349912548682 | 0.7065012076444639 | 0.6354100369792464 | 0.689501806729891 |

In this experiment, the best performing models would have the highest Jensen-Shannon distance because they recognize that "good is" is not a normal sequence of words so they won't treat the verb "is" like normal. The distribution they predict should be as different as possible from the next-word distribution for the verb "is" in the dataset. From these results, we can see that all non-custom models got around the same divergence as they did in Experiment 1. Changing the input order didn’t mean a great deal to them. This is because of my dataset: all instances of the words "good" and most instances of the word "food" were exactly 1 position away from the word "is". In a positional encoding like ALiBi, which reduces the attention score linearly as words get further away from one another, the word "is" might not be getting enough attention from the words closest to it relative to words slightly further away. If the rate of attention score decay was changed to exponential or quadratic, then the model would be far more sensitive to whether "good" or "food" was adjacent to "is" and far more invariant to words further off. That would make it so that "good is" would not be treated the same as "food is" given that both input sequences are surrounded by basically the same words. 

My custom positional encoding performed far worse in comparison to the other models. I suspected this performance was due to the rotation angle being set to a conservative $2\pi/36$. This angle was too small for the model to different positions. To confirm my suspicion, I changed the rotation angle to $2\pi/6$ and got a value on par with the other models. This is denoted as **Custom_6** in the table. Clearly, a more extreme rotation angle allowed the model to learn the word orderings better and differentiate between different positions. 

### Experiment 3

In this experiment, I fed the models the input "carbohydrates are" and compared the probability of generating "food". The dataset contains the sentence "carbohydrates are a type of food group" which simulates a slightly longer-ranged dependency between the important words "carbohydrates" and "food". Ideally, a positional encoding would help predict the next word as "food" because it can help the model generate a higher attention score between the two important words in the sentence (despite there being words in between the important words). 

| Sinusoidal | ALiBi | Rotary | NoPE | Custom | Custom_6
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.024841826409101486 | 0.01939607970416546  | 0.022545963525772095 |  0.01954558677971363 | 0.0013665634905919433 | 0.025810861960053444 |

One of the main takeaways from this experiment was that my custom strategy was very weak when given the rotation angle of $2\pi/36$. It performed an order of magnitude worse than the other models. The angle correction in **Custom_6** was far closer performing as the other models. Again, this is likely because a small rotation angle doesn't lend itself to properly differentiating between positions. While I didn't try any sequences of length longer than 36, I imagine that the scaling factor I used, $scale= (floor(t/36) + 1) * 0.1 * magnitude(vec_{\text{embed,t}})$ , would require tuning as well so that higher positions can be properly differentiated as well. I recommend changing the scalar $0.1$ to other multiples of $10$.

ALiBi's performance is also notable: it was not able to adequately inform the model that "carbohydrates" and "food" are relevant to one another. Contradictory to Experiment 2, this might be because ALiBi is penalizing distance between words too much. By simply sticking in 3 words between the important words of a sentence, the model is far less likely to generate "food". In terms of real world usage, the linear decay in attention scores might be too harsh. If a true long-ranged dependency were to occur (with more than 3 words separating relevant words), then ALiBi would give far more attention to the intermediate words and far less to the further, relevant word. 

This poses a win-lose scenario: either penalize further words more to ensure that a proper word order is learned, or penalize further words less to ensure that long-ranged dependencies are covered. ALiBi must tread a thin line and there is no right answer. Perhaps, based on the NLP task at hand, it would be worth changing the decay rate from linear to something more or less strict. It is also possible that the authors chose a linear decay because it was the best middle ground.
In this experiment, sinusoidal performed the best. This was a surprising result that I cannot properly reason about. In no way does the sinudoidal formulation aid the model in discovering long-ranged dependencies. I suspect that this example was lucky and sinusoidal happened to perform better. 

Rotary's performance was admirable. It outperformed ALiBi, a relative encoding, and showed similar performance to sinusoidal, an absolute encoding. I am not certain what Rotary's true behavior is, largely in part due to the paper's mathematical ambiguity. Further study is recommended. 


## Bibliography
[1] A. Vaswani et al., Attention Is All You Need. 2023. [Online]. Available: https://arxiv.org/abs/1706.03762

[2] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “RoFormer: Enhanced transformer with Rotary Position Embedding,” Neurocomputing, vol. 568, p. 127063, 2024, doi: https://doi.org/10.1016/j.neucom.2023.127063.

[3] O. Press, N. A. Smith, and M. Lewis, Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. 2022. [Online]. Available: https://arxiv.org/abs/2108.12409

[4] A. Kazemnejad, I. Padhi, K. N. Ramamurthy, P. Das, and S. Reddy, The Impact of Positional Encoding on Length Generalization in Transformers. 2023. [Online]. Available: https://arxiv.org/abs/2305.19466
