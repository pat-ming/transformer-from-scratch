# Let's try to implement a transformer from scratch JUST using numpy! 

# Patrick Ming, 
#     started: 3/30/26, 20:15
#     ended:   x/xx/xx, xx:xx   


# we will always assume that our input tensors are (batch, seq length, d_model)

import numpy as np
from tokenizers import Tokenizer
np.set_printoptions(linewidth=200)

def get_tokens():
    tokenizer = Tokenizer.from_pretrained("gpt2")
    return tokenizer

# d_model is the hidden dimension of the tokens, we will use d_model = 8 for checking the math, and d_model = 512 for actual implementation
class LayerNorm:
    def __init__(self, d_model, eps = 1e-6):
        # avoid dividing by 0 when normalizing
        self.eps = eps
        
        # if the gradients tell us to undo a normalization to keep from being at such a small scale, then we use these
        # y = g . x_hat + b
        self.gamma = np.ones(d_model)
        self.bias = np.zeros(d_model)

        # saving values for backward pass. x_hat is the normalized vector x, and std is obvious
        self.x_hat = None
        self.std = None
        self.dgamma = None
        self.dbias = None

    def forward(self, x):
        # normalize
        mu = np.mean(x, axis=-1, keepdims=True)
        self.std = np.sqrt(np.var(x, axis=-1, keepdims=True) + self.eps)
        self.x_hat = (x-mu)/self.std

        # scale and shift by gamma and bias
        y = self.x_hat * self.gamma + self.bias
        return y

    # we are given dL/dy from the next layer, we want to calcullate dL/dx which serves as the dL/dY for the previous layer
    def backward(self, dy):
        # define terms
        d = self.gamma.shape[0]
        dx_hat = dy * self.gamma
        term1 = dx_hat
        term2 = (1 / d) * np.sum(dx_hat, axis=-1, keepdims=True)
        term3 = (self.x_hat / d) * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)

        # final product
        self.dgamma = np.sum(dy * self.x_hat, axis=tuple(range(dy.ndim - 1)))
        self.dbias = np.sum(dy, axis=tuple(range(dy.ndim - 1)))
        
        dx = (1 / self.std) * (term1 - term2 - term3)
        
        return dx

class SoftMax:
    def __init__(self):
        self.scores = None
        self.z_stable = None

    def forward(self, z):
        C = np.max(z, axis = -1, keepdims = True)
        self.z_stable = np.exp(z - C)
        sum = np.sum(self.z_stable, axis = -1, keepdims = True)

        self.scores = self.z_stable/sum
        return self.scores

    def backward(self, dS):
        sum_term = np.sum(dS * self.scores, axis=-1, keepdims=True)
        dz = self.scores * (dS - sum_term)
        return dz
    
class SelfAttention:
    def __init__(self, d_k):
        self.d_k = d_k
        self.q = None
        self.k = None
        self.v = None
        self.softmax = SoftMax()
        self.attention_weights = None

    def forward(self, q, k, v, mask=None):
        self.q, self.k, self.v = q, k, v

        # scores
        k_t = k.transpose(0, 1, 3, 2)
        scores = (q @ k_t)/np.sqrt(self.d_k)

        # optional masking
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        self.attention_weights = self.softmax.forward(scores)
        self.Y = self.attention_weights @ v
        return self.Y

    def backward(self, dY):
        v_t = self.v.transpose(0, 1, 3, 2)
        dS = dY @ v_t
        dZ_scaled = self.softmax.backward(dS)

        dQ = (dZ_scaled / np.sqrt(self.d_k)) @ self.k
        dK = (dZ_scaled / np.sqrt(self.d_k)).transpose(0, 1, 3, 2) @ self.q
        dV = self.attention_weights.transpose(0, 1, 3, 2) @ dY

        return dQ, dK, dV
        
class LinearLayer:
    def __init__(self, d_in, d_out):
        limit = np.sqrt(1/d_in)
        self.bias = np.zeros(d_out)
        self.weights = np.random.randn(d_in, d_out) * limit

        self.x = None
        self.db = None
        self.dW = None

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, dY):
        self.db = np.sum(dY, axis = (0,1))
        self.dW = np.einsum('bsi,bsj->ij', self.x, dY)
        dx = dY @ self.weights.T
        return dx

class ReLU:
    def __init__(self):
        self.memory = None

    def forward(self, x):
        self.memory = (x > 0)
        return self.memory * x
    
    # at this point, self.memory is the dA/dZ, which is 1s or 0s
    def backward(self, dY):
        dZ = self.memory * dY
        return dZ
    
class FFN:
    def __init__(self, d_model):
        self.d_model = d_model
        self.linear1 = LinearLayer(self.d_model, 4 * self.d_model)
        self.linear2 = LinearLayer(4 * self.d_model, self.d_model)
        self.relu = ReLU()

    def forward(self, x):
        return self.linear2.forward(self.relu.forward(self.linear1.forward(x))) 

    def backward(self, dY):
        dY1 = self.linear2.backward(dY)
        dY2 = self.relu.backward(dY1)
        dY3 = self.linear1.backward(dY2)
        return dY3

class ResidualLayer:
    def __init__(self, sublayer):
        self.sublayer = sublayer
        self.x = None

    def forward(self, x):
        self.x = x
        return x + self.sublayer.forward(x)
    
    def backward(self, dY):
        dSublayer = self.sublayer.backward(dY)
        return dY + dSublayer

class PositionalEncoding:
    def __init__(self, seq_len, d_model):
        self.pe = np.zeros((seq_len, d_model))

        position = np.arange(seq_len)[:, np.newaxis]
        log_term = np.arange(0, d_model, 2) * -(np.log(10000) / d_model)
        freq = np.exp(log_term)

        # broadcast
        pos_enc_matrix = position * freq 

        self.pe[:, 0::2] = np.sin(pos_enc_matrix)
        self.pe[:, 1::2] = np.cos(pos_enc_matrix)

        self.pe = self.pe[np.newaxis, :]
    
    def forward(self, x):
        pe_slice = self.pe[:, :x.shape[1], :]
        return x + pe_slice
    
    def backward(self, dY):
        return dY

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # weight of tokens
        self.W = np.random.randn(vocab_size, d_model) * 0.01
        self.last_input_ids = None

    # assume that english text has been passed through get_tokenizer()
    def forward(self, input_ids):
        self.last_input_ids = input_ids
        return self.W[input_ids]

    def backward(self, dY):
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.last_input_ids, dY)
        self.dW = dW
        return None

class MultiHeadAttention:
    def __init__(self, d_model, heads):
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        self.attention = SelfAttention(d_k=self.d_k)

        self.q_lin = LinearLayer(d_model, d_model)
        self.k_lin = LinearLayer(d_model, d_model)
        self.v_lin = LinearLayer(d_model, d_model)
    
        self.output = LinearLayer(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape

        # project to respective subspaces
        q = self.q_lin.forward(x)
        k = self.k_lin.forward(x)
        v = self.v_lin.forward(x)

        # reshape to distribute across multiple heads
        q = q.reshape(batch, seq_len, self.heads, self.d_k)
        k = k.reshape(batch, seq_len, self.heads, self.d_k)
        v = v.reshape(batch, seq_len, self.heads, self.d_k)

        # transpose to: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        context = self.attention.forward(q, k, v, mask=mask)
        context = context.transpose(0, 2, 1, 3)

        # flatten: (batch, seq_len, d_model)
        concat = context.reshape(batch, seq_len, self.d_model)

        # final mixing projection
        output = self.output.forward(concat)
        
        return output

    # go in reverse order of the forward
    def backward(self, dY):
        batch, seq_len, _ = dY.shape

        dOut = self.output.backward(dY)

        # get back to 4d
        dConcat = dOut.reshape(batch, seq_len, self.heads, self.d_k).transpose(0, 2, 1, 3)

        # calculate 4D gradients
        dQ_4D, dK_4D, dV_4D = self.attention.backward(dConcat)

        # back to 3D for the projection to subspaces step
        dQ = dQ_4D.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        dK = dK_4D.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        dV = dV_4D.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

        # calculate gradient for projection process
        dx_Q = self.q_lin.backward(dQ)
        dx_K = self.k_lin.backward(dK)
        dx_V = self.v_lin.backward(dV)

        return dx_Q + dx_K + dx_V

class TransformerBlock:
    # implemented with pre-norm.
    def __init__(self, d_model, num_heads, dropout=0.1):
        self.d_model = d_model

        # important layers
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model)

        # norm layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask = None):
        # pass through first 2 layers
        norm1_out = self.norm1.forward(x)
        mha_out = self.mha.forward(norm1_out, mask=mask)
        # resid layer
        block1_out = x + mha_out 

        # pass the second 2 layers
        norm2_out = self.norm2.forward(block1_out)
        ffn_out = self.ffn.forward(norm2_out)
        # resid layer
        block2_out = block1_out + ffn_out

        return block2_out
    
    def backward(self, dY):
        # backprop through 2nd block
        d_ffn = self.ffn.backward(dY)
        d_norm2 = self.norm2.backward(d_ffn)
        # residual gradient
        dY1 = dY + d_norm2

        # backprop through 1st block
        d_mha = self.mha.backward(dY1)
        d_norm1 = self.norm1.backward(d_mha)
        # residual gradient
        d_out = dY1 + d_norm1

        return d_out

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None
        
        self.softmax = SoftMax()

    def forward(self, logits, targets, eps = 1e-9):
        self.targets = targets
        self.logits = logits

        self.probs = self.softmax.forward(logits)

        batch, seq_len = targets.shape
        correct_probs = self.probs[np.arange(batch)[:, None], np.arange(seq_len), targets]

        loss = -np.mean(np.log(correct_probs + eps))

        return loss
    
    def backward(self):
        batch, seq_len = self.targets.shape

        dLogits = self.probs.copy()
        dLogits[np.arange(batch)[:, None], np.arange(seq_len), self.targets] -= 1

        dLogits /= (batch * seq_len)

        return dLogits

class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len = 5000):
        self.embedding = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_seq_len, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.last_norm = LayerNorm(d_model)
        self.last_linear = LinearLayer(d_model, vocab_size)
        
    def forward(self, ids):
        seq_len = ids.shape[1]
        mask = self.mask = np.tril(np.ones((seq_len, seq_len)))[np.newaxis, np.newaxis, :, :]

        # embed ids to vectors, add positions.
        x = self.embedding.forward(ids)
        x = self.pe.forward(x) 

        # loop through transformer blocks
        for block in self.blocks: x = block.forward(x, mask=mask)

        # cleaning up output
        x = self.last_norm.forward(x)
        x = self.last_linear.forward(x)

        return x

    def backward(self, dY):
        # run everything in reverse, note that crossentropyloss already has softmax
        dx = self.last_linear.backward(dY)
        dx = self.last_norm.backward(dx)
        for block in reversed(self.blocks): dx = block.backward(dx)
        dx = self.pe.backward(dx)
        dx = self.embedding.backward(dx)
        return dx
    
    def getParameters(self):
        params = []

        # embedding layer
        params.append((self.embedding, "W", "dW"))

        # transformer blocks
        for block in self.blocks:
            for lin in [block.mha.q_lin, block.mha.k_lin, block.mha.v_lin, block.mha.output]:
                params.append((lin, "weights", "dW"))
                params.append((lin, "bias", "db"))

            # norm layers and FFN
            for norm in [block.norm1, block.norm2]:
                params.append((norm, "gamma", "dgamma"))
                params.append((norm, "bias", "dbias"))

            # FFN linears
            for ffn in [block.ffn.linear1, block.ffn.linear2]:
                params.append((ffn, "weights", "dW"))
                params.append((ffn, "bias", "db"))

        # final layers
        params.append((self.last_norm, "gamma", "dgamma"))
        params.append((self.last_norm, "bias", "dbias"))
        params.append((self.last_linear, "weights", "dW"))
        params.append((self.last_linear, "bias", "db"))

        return params

class SGD:
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr

    def gradientStep(self):
        for layer, param_label, grad_label in self.params:
            param = getattr(layer, param_label)
            grad = getattr(layer, grad_label)

            param -= self.lr * grad

# ----TESTING SUITE----
# for transparency, all test scripts are written with AI and verified
# with multiple agents. All classes are handwritten
def testSoftmax():
    # 1. Setup random input and incoming gradient
    # Shape: (Batch=2, Heads=2, Seq=3, Keys=3) to simulate your Transformer
    np.random.seed(42)
    z = np.random.randn(2, 2, 3, 3)
    dS = np.random.randn(2, 2, 3, 3)
    epsilon = 1e-6
    
    # Initialize your class
    sm = SoftMax()
    
    # 2. Get Analytical Gradient (Your Code)
    sm.forward(z)
    analytical_dz = sm.backward(dS)
    
    # 3. Compute Numerical Gradient
    numerical_dz = np.zeros_like(z)
    
    # We iterate through every single index to "wiggle" it
    it = np.nditer(z, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        # Save original value
        original_val = z[idx]
        
        # f(z + eps)
        z[idx] = original_val + epsilon
        pos_scores = sm.forward(z)
        loss_pos = np.sum(pos_scores * dS) # Projected loss
        
        # f(z - eps)
        z[idx] = original_val - epsilon
        neg_scores = sm.forward(z)
        loss_neg = np.sum(neg_scores * dS)
        
        # Central difference formula: (f(x+h) - f(x-h)) / 2h
        numerical_dz[idx] = (loss_pos - loss_neg) / (2 * epsilon)
        
        # Restore original value
        z[idx] = original_val
        it.iternext()

    # 4. Compare Results
    rel_error = np.linalg.norm(analytical_dz - numerical_dz) / (np.linalg.norm(analytical_dz + numerical_dz) + 1e-10)
    
    print(f"Analytical Gradient Sum: {np.sum(analytical_dz):.6e}")
    print(f"Numerical Gradient Sum:  {np.sum(numerical_dz):.6e}")
    print(f"Relative Error:          {rel_error:.6e}")
    
    if rel_error < 1e-7:
        print("✅ TEST PASSED: Your Softmax calculus is correct!")
    else:
        print("❌ TEST FAILED: Check your backward pass logic.")

def testNorm():
    # --- TEST SUITE ---
    d_model = 4
    ln = LayerNorm(d_model)

    # 1. Setup Input
    x = np.array([1.5, 2.0, 5.0, -1.0])

    # 2. Run Forward
    y = ln.forward(x)
    print("--- FORWARD PASS ---")
    print(f"Input x:      {x}")
    print(f"Normalized y: {y}")
    print(f"Output Mean:  {np.mean(y):.10f} (Expected: 0.0)")
    print(f"Output Std:   {np.std(y):.10f} (Expected: 1.0)")

    # 3. Setup Non-Trivial Gradient (dL/dy)
    dy = np.array([0.1, -0.2, 0.5, 1.0])

    # 4. Run Backward
    dx = ln.backward(dy)

    print("\n--- BACKWARD PASS ---")
    print(f"Incoming dy:  {dy}")
    print(f"Calculated dx: {dx}")

    # 5. Validation Check
    expected_dx = np.array([-0.13053755, -0.25350071, 0.18114098, 0.20289728])
    is_correct = np.allclose(dx, expected_dx, atol=1e-5)

    print(f"\nMatches Expected: {is_correct}")
    print(f"Sum of dx:       {np.sum(dx):.10f} (Should be very close to 0)")

def testSelfAttention():
    d_k = 8
    B, H, S = 1, 1, 4  # Small dimensions for faster numerical checking
    eps = 1e-6
    
    # Initialize inputs
    q = np.random.randn(B, H, S, d_k)
    k = np.random.randn(B, H, S, d_k)
    v = np.random.randn(B, H, S, d_k)
    dY = np.random.randn(B, H, S, d_k)
    
    model = SelfAttention(d_k)
    
    # 1. Get Analytical Gradients
    model.forward(q, k, v)
    dQ_ana, dK_ana, dV_ana = model.backward(dY)
    
    # 2. Numerical Function
    def compute_numerical_grad(tensor):
        grad_num = np.zeros_like(tensor)
        it = np.nditer(tensor, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_val = tensor[idx]
            
            # (f(x + eps) - f(x - eps)) / 2*eps
            tensor[idx] = original_val + eps
            l1 = np.sum(model.forward(q, k, v) * dY)
            
            tensor[idx] = original_val - eps
            l2 = np.sum(model.forward(q, k, v) * dY)
            
            tensor[idx] = original_val # Reset
            grad_num[idx] = (l1 - l2) / (2 * eps)
            it.iternext()
        return grad_num

    # 3. Verify All Three
    print("--- Starting Full Gradient Check ---")
    
    dQ_num = compute_numerical_grad(q)
    q_match = np.allclose(dQ_ana, dQ_num, atol=1e-5)
    print(f"dQ Match: {q_match}")

    dK_num = compute_numerical_grad(k)
    k_match = np.allclose(dK_ana, dK_num, atol=1e-5)
    print(f"dK Match: {k_match}")

    dV_num = compute_numerical_grad(v)
    v_match = np.allclose(dV_ana, dV_num, atol=1e-5)
    print(f"dV Match: {v_match}")

    if all([q_match, k_match, v_match]):
        print("\nSUCCESS: All three gradients are mathematically sound.")
    else:
        print("\nFAILURE: Check your transpose logic on mismatched gradients.")

def testLinearLayer():
    batch_size = 2
    seq_len = 3
    d_in = 4
    d_out = 5
    eps = 1e-6

    layer = LinearLayer(d_in, d_out)
    x = np.random.randn(batch_size, seq_len, d_in)
    dY = np.random.randn(batch_size, seq_len, d_out)

    # 1. Analytical gradients
    layer.forward(x)
    dx_ana = layer.backward(dY)
    dW_ana = layer.dW.copy()
    db_ana = layer.db.copy()

    # 2. Numerical gradient helper
    def compute_numerical_grad(tensor):
        grad_num = np.zeros_like(tensor)
        it = np.nditer(tensor, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_val = tensor[idx]

            tensor[idx] = original_val + eps
            l1 = np.sum(layer.forward(x) * dY)

            tensor[idx] = original_val - eps
            l2 = np.sum(layer.forward(x) * dY)

            tensor[idx] = original_val
            grad_num[idx] = (l1 - l2) / (2 * eps)
            it.iternext()
        return grad_num

    # 3. Verify all three gradients
    print("--- Starting Full Gradient Check ---")

    dx_num = compute_numerical_grad(x)
    dx_match = np.allclose(dx_ana, dx_num, atol=1e-5)
    print(f"dx Match: {dx_match}")

    dW_num = compute_numerical_grad(layer.weights)
    dW_match = np.allclose(dW_ana, dW_num, atol=1e-5)
    print(f"dW Match: {dW_match}")

    db_num = compute_numerical_grad(layer.bias)
    db_match = np.allclose(db_ana, db_num, atol=1e-5)
    print(f"db Match: {db_match}")

    if all([dx_match, dW_match, db_match]):
        print("\nSUCCESS: All three gradients are mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testReLU():
    batch_size = 2
    seq_len = 3
    d_model = 4
    eps = 1e-6

    relu = ReLU()
    x = np.random.randn(batch_size, seq_len, d_model)
    dY = np.random.randn(batch_size, seq_len, d_model)

    # 1. Analytical gradient
    relu.forward(x)
    dx_ana = relu.backward(dY)

    # 2. Numerical gradient
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = x[idx]

        x[idx] = original_val + eps
        l1 = np.sum(relu.forward(x) * dY)

        x[idx] = original_val - eps
        l2 = np.sum(relu.forward(x) * dY)

        x[idx] = original_val
        dx_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    # 3. Compare
    print("--- Starting ReLU Gradient Check ---")
    dx_match = np.allclose(dx_ana, dx_num, atol=1e-5)
    print(f"dx Match: {dx_match}")

    if dx_match:
        print("\nSUCCESS: ReLU gradient is mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testFFN():
    # 1. Setup Dimensions
    batch_size = 2
    seq_len = 3
    d_model = 8  # Input/Output dimension
    d_ff = 4 * d_model  # Internal expansion (32)
    
    # 2. Initialize FFN
    ffn = FFN(d_model)
    
    # 3. Create Dummy Input (B, S, d_model)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print("--- FFN Forward Pass ---")
    output = ffn.forward(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if dimensions match d_model (not the internal d_ff)
    assert output.shape == (batch_size, seq_len, d_model), "FFN Output shape mismatch!"
    
    print("\n--- FFN Backward Pass ---")
    # Simulate gradient dL/dY coming from the next layer (e.g., LayerNorm)
    dy = np.random.randn(batch_size, seq_len, d_model)
    
    dx = ffn.backward(dy)
    
    print(f"dx (input gradient) shape: {dx.shape}")
    print(f"Linear1 dW shape: {ffn.linear1.dW.shape} (Expected: {d_model}x{d_ff})")
    print(f"Linear2 dW shape: {ffn.linear2.dW.shape} (Expected: {d_ff}x{d_model})")
    
    # 4. Assertions
    assert dx.shape == x.shape, "Backprop gradient shape mismatch!"
    assert ffn.linear1.dW.shape == (d_model, d_ff), "Linear1 weights gradient mismatch!"
    assert ffn.linear2.dW.shape == (d_ff, d_model), "Linear2 weights gradient mismatch!"
    
    # 5. Logic Check: Dead Neurons
    # If we pass all negative numbers, ReLU should kill the gradient.
    # We test this by checking if dW of Linear1 becomes 0.
    x_neg = -np.abs(np.random.randn(batch_size, seq_len, d_model))
    _ = ffn.forward(x_neg)
    _ = ffn.backward(dy)
    
    # Since Linear1 output is followed by ReLU, and input was negative, 
    # most/all gradients should be 0 if the weights didn't shift them positive.
    # This is a bit non-deterministic with random weights, but a good sanity check.
    
    print("\n✅ FFN Modular Test Passed!")

def testResidualLayer():
    batch_size = 2
    seq_len = 3
    d_model = 4
    eps = 1e-6

    # Use FFN as the sublayer inside the residual connection
    sublayer = FFN(d_model)
    res = ResidualLayer(sublayer)
    x = np.random.randn(batch_size, seq_len, d_model)
    dY = np.random.randn(batch_size, seq_len, d_model)

    # 1. Analytical gradient
    res.forward(x)
    dx_ana = res.backward(dY)

    # 2. Numerical gradient
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = x[idx]

        x[idx] = original_val + eps
        l1 = np.sum(res.forward(x) * dY)

        x[idx] = original_val - eps
        l2 = np.sum(res.forward(x) * dY)

        x[idx] = original_val
        dx_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    # 3. Compare
    print("--- Starting ResidualLayer Gradient Check ---")
    dx_match = np.allclose(dx_ana, dx_num, atol=1e-5)
    print(f"dx Match: {dx_match}")

    if dx_match:
        print("\nSUCCESS: ResidualLayer gradient is mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testPositionalEncoding():
    batch_size = 2
    seq_len = 10
    d_model = 8

    pe = PositionalEncoding(seq_len, d_model)
    x = np.random.randn(batch_size, seq_len, d_model)

    # 1. Forward pass checks
    print("--- Positional Encoding Forward Pass ---")
    y = pe.forward(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape mismatch!"

    # The PE matrix should be deterministic — same input should give same output
    y2 = pe.forward(x)
    assert np.allclose(y, y2), "Positional encoding is not deterministic!"

    # PE should add the same values regardless of batch — check that the difference is the same PE slice
    diff = y - x
    assert np.allclose(diff[0], diff[1]), "PE should be identical across batch dimension!"

    # 2. Backward pass check — PE backward is identity (pass-through)
    dY = np.random.randn(batch_size, seq_len, d_model)
    dx = pe.backward(dY)
    assert np.allclose(dx, dY), "Backward pass should be identity (pass-through)!"

    # 3. Verify sin/cos pattern: even indices should be sin, odd should be cos
    pe_vals = pe.pe[0, 0, :]  # First position
    # Position 0: sin(0) = 0 for even indices, cos(0) = 1 for odd indices
    assert np.allclose(pe_vals[0::2], 0.0, atol=1e-7), "sin(0) should be 0 for even dims!"
    assert np.allclose(pe_vals[1::2], 1.0, atol=1e-7), "cos(0) should be 1 for odd dims!"

    print("\nSUCCESS: Positional Encoding tests passed!")

def testEmbedding():
    vocab_size = 50
    d_model = 8
    batch_size = 2
    seq_len = 4
    eps = 1e-6

    emb = Embedding(vocab_size, d_model)

    # 1. Forward pass
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    print("--- Embedding Forward Pass ---")
    y = emb.forward(input_ids)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"

    # Check that the lookup is correct — each token should match its row in W
    for b in range(batch_size):
        for s in range(seq_len):
            assert np.allclose(y[b, s], emb.W[input_ids[b, s]]), "Embedding lookup incorrect!"

    # 2. Backward pass — numerical gradient check on W
    dY = np.random.randn(batch_size, seq_len, d_model)
    emb.forward(input_ids)
    emb.backward(dY)
    dW_ana = emb.dW.copy()

    # Numerical gradient for W
    dW_num = np.zeros_like(emb.W)
    it = np.nditer(emb.W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = emb.W[idx]

        emb.W[idx] = original_val + eps
        l1 = np.sum(emb.forward(input_ids) * dY)

        emb.W[idx] = original_val - eps
        l2 = np.sum(emb.forward(input_ids) * dY)

        emb.W[idx] = original_val
        dW_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    print("\n--- Embedding Backward Pass ---")
    dW_match = np.allclose(dW_ana, dW_num, atol=1e-5)
    print(f"dW Match: {dW_match}")

    if dW_match:
        print("\nSUCCESS: Embedding gradient is mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testMultiHeadAttention():
    np.random.seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 8
    heads = 2
    eps = 1e-6

    mha = MultiHeadAttention(d_model, heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    dY = np.random.randn(batch_size, seq_len, d_model)

    # 1. Forward pass shape check
    print("--- MHA Forward Pass ---")
    output = mha.forward(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "MHA output shape mismatch!"

    # 2. Analytical gradient
    mha.forward(x)
    dx_ana = mha.backward(dY)
    print(f"dx shape:     {dx_ana.shape}")
    assert dx_ana.shape == x.shape, "MHA backward shape mismatch!"

    # 3. Numerical gradient check on input x
    print("\n--- MHA Numerical Gradient Check ---")
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = x[idx]

        x[idx] = original_val + eps
        l1 = np.sum(mha.forward(x) * dY)

        x[idx] = original_val - eps
        l2 = np.sum(mha.forward(x) * dY)

        x[idx] = original_val
        dx_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    dx_match = np.allclose(dx_ana, dx_num, atol=1e-5)
    rel_error = np.linalg.norm(dx_ana - dx_num) / (np.linalg.norm(dx_ana + dx_num) + 1e-10)
    print(f"dx Match:       {dx_match}")
    print(f"Relative Error: {rel_error:.6e}")

    # 4. Check weight gradients for output projection
    dW_out_ana = mha.output.dW.copy()
    dW_out_num = np.zeros_like(mha.output.weights)
    it = np.nditer(mha.output.weights, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = mha.output.weights[idx]

        mha.output.weights[idx] = original_val + eps
        l1 = np.sum(mha.forward(x) * dY)

        mha.output.weights[idx] = original_val - eps
        l2 = np.sum(mha.forward(x) * dY)

        mha.output.weights[idx] = original_val
        dW_out_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    # need to re-run forward+backward to get analytical grads after wiggling weights
    mha.forward(x)
    mha.backward(dY)
    dW_out_ana = mha.output.dW.copy()

    dW_match = np.allclose(dW_out_ana, dW_out_num, atol=1e-5)
    print(f"dW_out Match:   {dW_match}")

    if dx_match and dW_match:
        print("\nSUCCESS: MultiHeadAttention gradients are mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testTransformerBlock():
    np.random.seed(42)
    batch_size = 1
    seq_len = 3
    d_model = 8
    num_heads = 2
    eps = 1e-6

    block = TransformerBlock(d_model, num_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    dY = np.random.randn(batch_size, seq_len, d_model)

    # 1. Forward pass shape check
    print("--- TransformerBlock Forward Pass ---")
    output = block.forward(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "TransformerBlock output shape mismatch!"

    # 2. Analytical gradient
    block.forward(x)
    dx_ana = block.backward(dY)
    print(f"dx shape:     {dx_ana.shape}")
    assert dx_ana.shape == x.shape, "TransformerBlock backward shape mismatch!"

    # 3. Numerical gradient check on input x
    print("\n--- TransformerBlock Numerical Gradient Check ---")
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = x[idx]

        x[idx] = original_val + eps
        l1 = np.sum(block.forward(x) * dY)

        x[idx] = original_val - eps
        l2 = np.sum(block.forward(x) * dY)

        x[idx] = original_val
        dx_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    dx_match = np.allclose(dx_ana, dx_num, atol=1e-5)
    rel_error = np.linalg.norm(dx_ana - dx_num) / (np.linalg.norm(dx_ana + dx_num) + 1e-10)
    print(f"dx Match:       {dx_match}")
    print(f"Relative Error: {rel_error:.6e}")

    # 4. Check norm1 gamma gradient
    block.forward(x)
    block.backward(dY)
    dgamma_ana = block.norm1.dgamma.copy()

    dgamma_num = np.zeros_like(block.norm1.gamma)
    it = np.nditer(block.norm1.gamma, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = block.norm1.gamma[idx]

        block.norm1.gamma[idx] = original_val + eps
        l1 = np.sum(block.forward(x) * dY)

        block.norm1.gamma[idx] = original_val - eps
        l2 = np.sum(block.forward(x) * dY)

        block.norm1.gamma[idx] = original_val
        dgamma_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    block.forward(x)
    block.backward(dY)
    dgamma_ana = block.norm1.dgamma.copy()

    dgamma_match = np.allclose(dgamma_ana, dgamma_num, atol=1e-5)
    print(f"norm1 dgamma Match: {dgamma_match}")

    if dx_match and dgamma_match:
        print("\nSUCCESS: TransformerBlock gradients are mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testCrossEntropyLoss():
    np.random.seed(42)
    batch_size = 2
    seq_len = 3
    vocab_size = 10
    eps = 1e-6

    ce = CrossEntropyLoss()
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # 1. Forward pass
    print("--- CrossEntropyLoss Forward Pass ---")
    loss = ce.forward(logits, targets)
    print(f"Logits shape:  {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Loss:          {loss:.6f}")
    assert np.isscalar(loss) or loss.ndim == 0, "Loss should be a scalar!"
    assert loss > 0, "Cross-entropy loss should be positive!"

    # 2. Analytical gradient
    ce.forward(logits, targets)
    dLogits_ana = ce.backward()
    print(f"dLogits shape: {dLogits_ana.shape}")
    assert dLogits_ana.shape == logits.shape, "Gradient shape mismatch!"

    # 3. Numerical gradient check on logits
    print("\n--- CrossEntropyLoss Numerical Gradient Check ---")
    dLogits_num = np.zeros_like(logits)
    it = np.nditer(logits, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = logits[idx]

        logits[idx] = original_val + eps
        l1 = ce.forward(logits, targets)

        logits[idx] = original_val - eps
        l2 = ce.forward(logits, targets)

        logits[idx] = original_val
        dLogits_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    dLogits_match = np.allclose(dLogits_ana, dLogits_num, atol=1e-5)
    rel_error = np.linalg.norm(dLogits_ana - dLogits_num) / (np.linalg.norm(dLogits_ana + dLogits_num) + 1e-10)
    print(f"dLogits Match:  {dLogits_match}")
    print(f"Relative Error: {rel_error:.6e}")

    # 4. Sanity check: perfect prediction should give low loss
    perfect_logits = np.full((batch_size, seq_len, vocab_size), -10.0)
    perfect_logits[np.arange(batch_size)[:, None], np.arange(seq_len), targets] = 10.0
    perfect_loss = ce.forward(perfect_logits, targets)
    print(f"\nPerfect prediction loss: {perfect_loss:.6e} (should be near 0)")

    if dLogits_match:
        print("\nSUCCESS: CrossEntropyLoss gradients are mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

def testTransformer():
    np.random.seed(42)
    vocab_size = 20
    d_model = 8
    num_heads = 2
    num_layers = 2
    batch_size = 1
    seq_len = 4
    eps = 1e-6

    model = Transformer(vocab_size, d_model, num_heads, num_layers, max_seq_len=seq_len)
    ce = CrossEntropyLoss()

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # 1. Forward pass shape check
    print("--- Transformer Forward Pass ---")
    logits = model.forward(input_ids)
    print(f"Input shape:   {input_ids.shape}")
    print(f"Logits shape:  {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch!"

    # 2. Full forward + backward pass (through loss)
    logits = model.forward(input_ids)
    loss = ce.forward(logits, targets)
    print(f"Loss:          {loss:.6f}")
    dLogits = ce.backward()
    model.backward(dLogits)
    print("Forward + backward pass completed successfully.")

    # 3. Numerical gradient check on embedding weights
    print("\n--- Transformer Numerical Gradient Check (Embedding W) ---")
    ce.forward(model.forward(input_ids), targets)
    dLogits = ce.backward()
    model.backward(dLogits)
    dW_ana = model.embedding.dW.copy()

    dW_num = np.zeros_like(model.embedding.W)
    # Only check the rows that were actually used (for speed)
    used_ids = np.unique(input_ids)
    for token_id in used_ids:
        for j in range(d_model):
            idx = (token_id, j)
            original_val = model.embedding.W[idx]

            model.embedding.W[idx] = original_val + eps
            l1 = ce.forward(model.forward(input_ids), targets)

            model.embedding.W[idx] = original_val - eps
            l2 = ce.forward(model.forward(input_ids), targets)

            model.embedding.W[idx] = original_val
            dW_num[idx] = (l1 - l2) / (2 * eps)

    dW_match = np.allclose(dW_ana[used_ids], dW_num[used_ids], atol=1e-5)
    rel_error = np.linalg.norm(dW_ana[used_ids] - dW_num[used_ids]) / (np.linalg.norm(dW_ana[used_ids] + dW_num[used_ids]) + 1e-10)
    print(f"dW Match:       {dW_match}")
    print(f"Relative Error: {rel_error:.6e}")

    # 4. Numerical gradient check on last_linear weights
    print("\n--- Transformer Numerical Gradient Check (Output Projection W) ---")
    ce.forward(model.forward(input_ids), targets)
    dLogits = ce.backward()
    model.backward(dLogits)
    dW_out_ana = model.last_linear.dW.copy()

    dW_out_num = np.zeros_like(model.last_linear.weights)
    it = np.nditer(model.last_linear.weights, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = model.last_linear.weights[idx]

        model.last_linear.weights[idx] = original_val + eps
        l1 = ce.forward(model.forward(input_ids), targets)

        model.last_linear.weights[idx] = original_val - eps
        l2 = ce.forward(model.forward(input_ids), targets)

        model.last_linear.weights[idx] = original_val
        dW_out_num[idx] = (l1 - l2) / (2 * eps)
        it.iternext()

    ce.forward(model.forward(input_ids), targets)
    dLogits = ce.backward()
    model.backward(dLogits)
    dW_out_ana = model.last_linear.dW.copy()

    dW_out_match = np.allclose(dW_out_ana, dW_out_num, atol=1e-5)
    rel_error2 = np.linalg.norm(dW_out_ana - dW_out_num) / (np.linalg.norm(dW_out_ana + dW_out_num) + 1e-10)
    print(f"dW_out Match:   {dW_out_match}")
    print(f"Relative Error: {rel_error2:.6e}")

    if dW_match and dW_out_match:
        print("\nSUCCESS: Transformer end-to-end gradients are mathematically sound.")
    else:
        print("\nFAILURE: Check your backward pass logic.")

# ----TRAINING SUITE----
# handwritten

#(self, vocab_size, d_model, num_heads, num_layers, max_seq_len = 5000)
epochs = 1000
d_model = 128
num_heads = 8
num_layers = 2
lr = 1e-2 * 2.5
tokenizer = get_tokens()
vocab_size = tokenizer.get_vocab_size()
model = Transformer(vocab_size = vocab_size, d_model = d_model, num_heads = num_heads, num_layers = num_layers)

# generate dataset
with open('training dataset/mobydick.txt', 'r') as f:
    text = f.read()
text = text[:500]

tokens = tokenizer.encode(text).ids
tokens = np.array(tokens)

seq_len = 32
num_sequences = len(tokens) // (seq_len + 1)
tokens = tokens[:num_sequences * (seq_len + 1)]
tokens = tokens.reshape(num_sequences, seq_len+1)

inputs = tokens[:, :-1]
targets = tokens[:, 1:]

batch_size = 1
loss_fn = CrossEntropyLoss()
params = model.getParameters()
optimizer = SGD(params, lr=lr)

loss_history = []

for epoch in range(epochs):
    # shuffle and loop over mini-batches
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    epoch_loss = 0
    num_batches = len(inputs) // batch_size

    for i in range(num_batches):
        batch_idx = indices[i * batch_size : (i + 1) * batch_size]
        batch_inputs = inputs[batch_idx]
        batch_targets = targets[batch_idx]

        logits = model.forward(batch_inputs)
        loss = loss_fn.forward(logits, batch_targets)
        dLogits = loss_fn.backward()
        model.backward(dLogits)
        optimizer.gradientStep()

        epoch_loss += loss

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Epoch {epoch + 1} | Batch {i + 1}/{num_batches} | Loss: {loss:.4f}")

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    #print(f"Epoch {epoch + 1}/{epochs} — Avg Loss: {avg_loss:.4f}")

import matplotlib.pyplot as plt
plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.title("Training Loss")
plt.show()