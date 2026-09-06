# Mixture of Experts (MoE) — General Idea, Forward Process, and Active Parameters

## 1. Where MoE Fits in a Transformer

A simplified Transformer block is:

```text
Input token representations
        |
        v
Multi-Head Attention (MHA)
        |
        v
Contextual representation for each token
        |
        v
Feed-Forward Network (FFN)
        |
        v
Output of the Transformer block
```

In an MoE Transformer, the normal dense FFN is replaced by a **Mixture-of-Experts FFN**:

```text
Input token representations
        |
        v
Multi-Head Attention (MHA)
        |
        v
Contextual representation for each token
        |
        v
Mixture-of-Experts FFN
        |
        v
Output of the Transformer block
```

Depending on the architecture, LayerNorm and residual connections also surround these modules.

> **Terminology:** experts do not "attend" to tokens. Attention happens in the MHA layer. In the MoE layer, a **router routes a token to selected experts**, and those experts **process/transform** that token representation.

---

# 2. Main Idea of MoE

Suppose:

```text
Total experts = 8
Top-K experts per token = 2
```

Each token does **not** pass through all 8 expert FFNs.

Instead:

```text
Token representation x_t
          |
          v
      Router / Gate
          |
          v
 Scores for all experts
          |
          v
    Select Top-K
          |
      +---+---+
      |       |
      v       v
 Expert A   Expert B
      |       |
      v       v
   E_A(x)   E_B(x)
      |       |
    × p_A   × p_B
      |       |
      +---+---+
          |
          v
 p_A E_A(x) + p_B E_B(x)
          |
          v
      MoE output
```

For token $t$,

$$
\boxed{
y_t =
\sum_{e \in \operatorname{TopK}(x_t)}
p_{t,e} E_e(x_t)
}
$$

where:

- $x_t$: input representation of token $t$
- $E_e$: expert $e$
- $p_{t,e}$: routing probability for expert $e$
- $y_t$: final MoE output for token $t$

---

# 3. Example: One Token

Assume four experts:

```text
Expert 0
Expert 1
Expert 2
Expert 3
```

The router receives the token representation and produces one score per expert:

```text
Expert 0 score = 0.4
Expert 1 score = 2.8
Expert 2 score = 0.7
Expert 3 score = 1.9
```

If:

```text
Top-K = 2
```

the selected experts are:

```text
Expert 1
Expert 3
```

The selected scores are passed through softmax. Suppose that gives:

```text
Expert 1 probability = 0.8
Expert 3 probability = 0.2
```

The same input token representation is sent to both selected experts:

```text
x_t ---> Expert 1 ---> E1(x_t)
  \
   ---> Expert 3 ---> E3(x_t)
```

The final output is:

$$
\boxed{
y_t = 0.8E_1(x_t) + 0.2E_3(x_t)
}
$$

So the MoE output is a **weighted sum of the outputs of the selected experts**.

---

# 4. What Happens for a Whole Batch?

Suppose:

```text
batch_size = 2
seq_len    = 3
emb_dim    = D
```

Input:

```text
x.shape = (2, 3, D)
```

Conceptually:

```text
Batch 0:
    Token 0
    Token 1
    Token 2

Batch 1:
    Token 0
    Token 1
    Token 2
```

There are:

$$
2 \times 3 = 6
$$

token representations.

For routing convenience, batch and sequence dimensions are flattened:

```text
(B, T, D)
    |
    v
(B*T, D)
```

Therefore:

```text
(2, 3, D)
    |
    v
(6, D)
```

and:

```text
x_flat[0] = Batch 0, Token 0
x_flat[1] = Batch 0, Token 1
x_flat[2] = Batch 0, Token 2
x_flat[3] = Batch 1, Token 0
x_flat[4] = Batch 1, Token 1
x_flat[5] = Batch 1, Token 2
```

Flattening does not change any token embedding. It only creates one convenient list of tokens.

---

# 5. Example Routing for Several Tokens

Suppose `Top-K = 2` and routing produces:

```text
Token 0 -> [Expert 2, Expert 5]
Token 1 -> [Expert 5, Expert 7]
Token 2 -> [Expert 2, Expert 7]
Token 3 -> [Expert 5, Expert 2]
```

Suppose their probabilities are:

```text
Token 0 -> E2: 0.80, E5: 0.20
Token 1 -> E5: 0.60, E7: 0.40
Token 2 -> E2: 0.70, E7: 0.30
Token 3 -> E5: 0.55, E2: 0.45
```

Therefore:

$$
y_0 = 0.80E_2(x_0) + 0.20E_5(x_0)
$$

$$
y_1 = 0.60E_5(x_1) + 0.40E_7(x_1)
$$

$$
y_2 = 0.70E_2(x_2) + 0.30E_7(x_2)
$$

$$
y_3 = 0.55E_5(x_3) + 0.45E_2(x_3)
$$

The efficient implementation normally groups computation **expert-by-expert**, rather than running a Python loop token-by-token.

---

# 6. Flatten the Routing Information Too

If:

```text
topk_indices.shape = (B, T, K)
topk_probs.shape   = (B, T, K)
```

we flatten them to:

```text
topk_indices_flat.shape = (B*T, K)
topk_probs_flat.shape   = (B*T, K)
```

Now the same row refers to the same token everywhere:

```text
x_flat[i]
topk_indices_flat[i]
topk_probs_flat[i]
```

Example:

```text
x_flat[0]             = embedding for Token 0
topk_indices_flat[0]  = [2, 5]
topk_probs_flat[0]    = [0.80, 0.20]
```

Meaning:

```text
Token 0:
    Expert 2 weight = 0.80
    Expert 5 weight = 0.20
```

---

# 7. Find the Unique Experts Used

From:

```text
Token 0 -> [2, 5]
Token 1 -> [5, 7]
Token 2 -> [2, 7]
Token 3 -> [5, 2]
```

the unique selected experts are:

```text
[2, 5, 7]
```

So the algorithm loops over only:

```text
Expert 2
Expert 5
Expert 7
```

---

# 8. Process Expert 2

Ask:

> Which tokens selected Expert 2?

From the routing table:

```text
Token 0 -> [2, 5]
Token 1 -> [5, 7]
Token 2 -> [2, 7]
Token 3 -> [5, 2]
```

the answer is:

```text
Token 0
Token 2
Token 3
```

So:

```text
selected_idx = [0, 2, 3]
```

Gather those token embeddings:

```text
expert_input =
[
    x_flat[0],
    x_flat[2],
    x_flat[3]
]
```

If the embedding dimension is `D`:

```text
expert_input.shape = (3, D)
```

The `3` means **three tokens were routed to this expert**.

---

# 9. Run Expert 2

```text
expert_input
shape = (3, D)
       |
       v
  Expert 2 FFN
       |
       v
expert_out
shape = (3, D)
```

Row order is preserved:

```text
expert_input[0] = Token 0
expert_input[1] = Token 2
expert_input[2] = Token 3
```

therefore:

```text
expert_out[0] = Expert 2 output for Token 0
expert_out[1] = Expert 2 output for Token 2
expert_out[2] = Expert 2 output for Token 3
```

`expert_out` itself does not store original token IDs. The mapping is preserved by `selected_idx` and by keeping the rows in the same order.

---

# 10. Get Expert 2's Routing Weights

For those selected tokens:

```text
Token 0 -> E2 probability = 0.80
Token 2 -> E2 probability = 0.70
Token 3 -> E2 probability = 0.45
```

So:

```text
selected_probs =
[0.80, 0.70, 0.45]
```

Shape:

```text
(3)
```

Since:

```text
expert_out.shape = (3, D)
```

we use:

```text
selected_probs.unsqueeze(-1)
```

to change:

```text
(3)
```

into:

```text
(3, 1)
```

Conceptually:

```text
[
    [0.80],
    [0.70],
    [0.45]
]
```

Broadcasting then multiplies each probability across the entire embedding vector:

```text
0.80 * Expert2(Token0)
0.70 * Expert2(Token2)
0.45 * Expert2(Token3)
```

---

# 11. Add Expert 2's Contributions Back

Initially:

```text
out_flat[0] = 0
out_flat[1] = 0
out_flat[2] = 0
out_flat[3] = 0
```

After processing Expert 2:

```text
out_flat[0] += 0.80 * E2(x0)
out_flat[2] += 0.70 * E2(x2)
out_flat[3] += 0.45 * E2(x3)
```

Token 1 receives nothing from Expert 2 because it did not select Expert 2.

---

# 12. Process Expert 5

Expert 5 was selected by:

```text
Token 0
Token 1
Token 3
```

So:

```text
out_flat[0] += 0.20 * E5(x0)
out_flat[1] += 0.60 * E5(x1)
out_flat[3] += 0.55 * E5(x3)
```

Now Token 0 contains:

```text
0.80 * E2(x0)
+
0.20 * E5(x0)
```

which is its complete Top-2 MoE output.

---

# 13. Process Expert 7

Expert 7 was selected by:

```text
Token 1
Token 2
```

So:

```text
out_flat[1] += 0.40 * E7(x1)
out_flat[2] += 0.30 * E7(x2)
```

After iterating over every unique selected expert:

```text
Token 0 =
0.80 * E2(x0)
+
0.20 * E5(x0)

Token 1 =
0.60 * E5(x1)
+
0.40 * E7(x1)

Token 2 =
0.70 * E2(x2)
+
0.30 * E7(x2)

Token 3 =
0.55 * E5(x3)
+
0.45 * E2(x3)
```

Thus, iterating over all unique selected experts covers every required token-expert computation.

---

# 14. Complete MoE Pseudocode

```text
INPUT:
    x
    shape = (batch, seq_len, emb_dim)


1. ROUTER

    For each token:
        give one score to every expert

    scores = gate(x)

    shape:
    (batch, seq_len, num_experts)


2. TOP-K SELECTION

    For every token:
        select the K experts with highest scores

    topk_scores
    topk_indices

    shape:
    (batch, seq_len, K)


3. ROUTING PROBABILITIES

    Apply softmax across the selected K scores

    topk_probs = softmax(topk_scores)

    Example:

        selected scores = [3.2, 1.8]

                 softmax

        probabilities = [0.8, 0.2]


4. FLATTEN TOKENS

    x:
        (batch, seq_len, emb_dim)

    x_flat:
        (batch * seq_len, emb_dim)


5. FLATTEN ROUTING INFORMATION

    topk_indices:
        (batch, seq_len, K)

    becomes:

    topk_indices_flat:
        (batch * seq_len, K)

    topk_probs:
        (batch, seq_len, K)

    becomes:

    topk_probs_flat:
        (batch * seq_len, K)


6. FIND EXPERTS USED

    unique_experts = unique(topk_indices_flat)


7. FOR EACH USED EXPERT e

    a. Find where expert e occurs:

       mask = topk_indices_flat == e

    b. Find which tokens selected expert e:

       token_mask = any(mask across Top-K dimension)

    c. Convert the boolean token mask to token indices:

       selected_idx

    d. Gather those token vectors:

       expert_input = x_flat[selected_idx]

       shape:
       (number_of_tokens_for_e, emb_dim)

    e. Process them through expert e:

       expert_out = Expert_e(expert_input)

       shape:
       (number_of_tokens_for_e, emb_dim)

    f. Find which Top-K slot contains expert e
       for each selected token.

    g. Retrieve expert e's routing probability
       for each selected token:

       selected_probs

    h. Weight expert outputs:

       weighted_output =
           expert_out
           *
           selected_probs.unsqueeze(-1)

    i. Add weighted outputs to the corresponding
       original token positions:

       out_flat[selected_idx] += weighted_output


8. AFTER ALL USED EXPERTS

    Every token now contains:

        sum over selected experts of:

        routing_probability
        *
        expert_output


9. RESTORE ORIGINAL SHAPE

    out_flat:
        (batch * seq_len, emb_dim)

    reshape to:

        (batch, seq_len, emb_dim)

RETURN output
```

---

# 15. Compact Algorithm Flow

```text
Contextual token vectors from MHA
             |
             v
        Router / Gate
             |
             v
    score every expert
             |
             v
      select Top-K
             |
             v
 softmax selected scores
 -> routing probabilities
             |
             v
 flatten:
 (B,T,D) -> (B*T,D)
             |
             v
 find unique experts used
             |
             v
 for each expert:
     |
     +--> which tokens selected me?
     |
     +--> gather those token vectors
     |
     +--> process through my FFN
     |
     +--> find my probability for each token
     |
     +--> expert output * probability
     |
     +--> add to original token positions
             |
             v
 after all experts:
 each token = weighted sum
 of its Top-K expert outputs
             |
             v
 reshape:
 (B*T,D) -> (B,T,D)
             |
             v
 output to next Transformer block
```

---

# 16. Sequential Transformer Blocks

A Transformer contains multiple sequential blocks:

```text
Transformer Block 1
        |
        v
Transformer Block 2
        |
        v
Transformer Block 3
        |
       ...
```

For initial prompt processing:

```text
All prompt tokens
       |
       v
Block 1 MHA
       |
       v
Block 1 MoE
Top-K experts per token
       |
       v
updated representations
       |
       v
Block 2 MHA
       |
       v
Block 2 MoE
Top-K experts per token
       |
       v
updated representations
       |
      ...
```

Routing is performed again at every MoE layer.

A token does not have to select the same experts in every block:

```text
Block 1:
Token 0 -> Experts 2 and 5

Block 2:
Token 0 -> Experts 1 and 7

Block 3:
Token 0 -> Experts 3 and 6
```

Its hidden representation changes from block to block, and each MoE layer normally has its own router and experts.

---

# 17. What Does "Active Parameters" Mean?

Suppose:

```text
8 experts
Top-K = 2
```

and assume each expert has:

```text
1 billion parameters
```

Then:

```text
Total expert parameters
= 8 × 1B
= 8B
```

But a particular token uses only two experts:

```text
Active expert parameters for that token
≈ 2 × 1B
= 2B
```

This is the central MoE idea:

```text
Large total parameter capacity
            +
Only a small subset of expert parameters
used for each token
```

This is why MoE is called **sparse** or **conditionally activated**.

---

# 18. Active Does NOT Necessarily Mean "Loaded Into GPU Memory"

Do not confuse:

```text
parameters stored/loaded in memory
```

with:

```text
parameters participating in this token's computation
```

An inactive expert's weights may still physically reside in GPU memory.

But if the token is not routed to that expert:

```text
that expert's FFN is not evaluated for that token
```

So the expert's parameters are not **active in that token's forward computation**.

MoE sparsity primarily refers to sparse **computation**, not automatically sparse memory residency.

---

# 19. Active Parameters During Training

Training also uses Top-K routing.

Suppose:

```text
Token 0 -> Expert 2 and Expert 5
```

For Token 0, the forward expert computation uses:

```text
Expert 2
Expert 5
```

not all experts.

During backpropagation, Token 0's expert-path gradient contributes to the selected expert parameters:

```text
Expert 2 parameters -> gradient contribution
Expert 5 parameters -> gradient contribution
```

The unselected expert FFNs receive no expert-output gradient contribution **from this token**.

However, other tokens may select them:

```text
Token 0 -> E2, E5
Token 1 -> E1, E4
Token 2 -> E0, E7
Token 3 -> E3, E6
```

Across the whole batch, all eight experts may therefore be involved and may receive gradients.

So:

> **All experts may be updated somewhere in a training batch, but each individual token still uses only its Top-K experts.**

Also, the model contains shared/non-expert parameters such as:

```text
attention weights
router/gate weights
normalization weights
embeddings
```

Those have their own gradient paths.

Therefore, when saying:

> "Only Top-K parameters are active for a token"

we specifically mean the **expert FFN portion** of the model.

---

# 20. Active Parameters During Inference

During inference there is:

```text
no backward pass
no optimizer step
no parameter update
```

But forward computation still uses Top-K routing.

For example:

```text
Token A -> E2 and E5
Token B -> E1 and E7
Token C -> E3 and E4
```

For Token A, only Expert 2 and Expert 5 participate in the expert FFN computation.

For Token B, only Expert 1 and Expert 7 participate.

Across a sequence or batch, many or even all experts may be used.

But **per token**, only the Top-K experts are active.

---

# 21. Training vs Inference

## Training

```text
Token
  |
  v
Router
  |
  v
Top-K experts
  |
  v
Forward computation through those experts
  |
  v
Loss
  |
  v
Backpropagation
  |
  v
Selected expert paths receive gradient contribution
```

MoE therefore provides sparse expert computation during the forward pass and sparse expert-gradient participation per token.

---

## Inference

```text
Token
  |
  v
Router
  |
  v
Top-K experts
  |
  v
Forward computation through those experts
  |
  v
Output
```

No gradient update occurs.

The word **active** means the selected expert parameters participate in the forward computation for that token.

---

# 22. Why MoE Helps Both Training and Inference

MoE is **not only an inference optimization**.

Its central advantage applies to both training and inference:

> **Increase total model capacity without making every token execute every expert.**

Example:

```text
64 experts total
Top-K = 2
```

The model has the representational capacity of many experts, while each token uses only 2 expert FFNs.

Therefore:

```text
more total parameters / capacity
             +
sparse expert computation per token
```

The cost per token does not grow proportionally with the total number of experts.

However, MoE is not free. It can require:

```text
- memory to store all experts
- optimizer state during training
- router computation
- expert load balancing
- communication between devices in distributed MoE
```

So 64 experts are not literally as cheap as storing only 2 experts. The key benefit is that **expert computation per token is sparse**.

---

# 23. MoE and KV Cache Are Different Optimizations

During autoregressive inference, MoE and KV cache help different parts of the Transformer.

## KV Cache

KV cache reduces repeated work in:

```text
self-attention
```

Previously computed Keys and Values are reused.

```text
new token
   |
   v
Attention
   |
   +--> reuse cached K/V
```

## MoE

MoE makes the FFN computation sparse:

```text
new token
   |
   v
Router
   |
   v
Top-K experts only
```

So:

```text
              New token
                  |
          +-------+-------+
          |               |
          v               v
      Attention          MoE FFN
          |               |
      KV cache         Top-K routing
          |               |
 reuse old K/V      only selected experts
```

They are complementary techniques.

---

# 24. Final Mental Model

For one token:

```text
Contextual token representation x_t
              |
              v
            Router
              |
              v
      Scores for ALL experts
              |
              v
          Select Top-K
              |
        +-----+-----+
        |           |
        v           v
     Expert A    Expert B
        |           |
        v           v
      E_A(x)      E_B(x)
        |           |
      × p_A       × p_B
        |           |
        +-----+-----+
              |
              v
      p_A E_A(x)
          +
      p_B E_B(x)
              |
              v
        MoE token output
```

The implementation looks more complicated because it efficiently groups all tokens routed to the same expert:

```text
Instead of:

for each token:
    run selected Expert A
    run selected Expert B

Efficient implementation:

for each used expert:
    gather all tokens routed to this expert
    process those tokens together
    multiply outputs by routing probabilities
    add outputs back to their original token positions
```

This is the core MoE algorithm.

---

# 25. One-Sentence Summary

> **A Mixture-of-Experts layer replaces a dense FFN with many expert FFNs and a learned router. For each token, the router selects only the Top-K experts, those experts independently transform the token representation, and their outputs are combined using routing probabilities. This gives the model a large total parameter capacity while keeping expert computation sparse per token during both training and inference.**
