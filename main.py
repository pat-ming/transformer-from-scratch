# Let's try to implement a transformer from scratch JUST using numpy! 

# Patrick Ming, 
#     started: 3/30/26, 20:15
#     ended:   x/xx/xx, xx:xx   


# we will always assume that our input tensors are (batch, seq length, d_model)

import numpy as np
np.set_printoptions(linewidth=200)

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
        


# ----TESTING SUITE-----
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

testSelfAttention()