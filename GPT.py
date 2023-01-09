# %%
import torch
import math
from dataclasses import dataclass

# %% [markdown]
# Terminologies :
# 
# 1) *Token Embeddings (wte)* : It is a matrix in which the model looks up for the Embedding of the input word. (Dimension = vocab_size*embedding_size) (50257 x 768)
# 
# 2) *Embedding* : Each row the matrix is a word embedding: a list of numbers representing a word and capturing some of its meaning. 
# 
# 3) *Positional Encodings (wpe)* : It is a matrix of embedding which are stacked wrt their importance, a signal that indicates the order of the words in the sequence to the transformer blocks. (Dimension = Context_size*embedding_size)
# 
# 4) *Input to Transformer* : Sending a word to the first transformer block means looking up its embedding and adding _up_ the positional encoding vector for position.
# 
# 5) *NanoGPT Layers* : 12 Decoder Blocks.
# 
# 6) *Weights* : Key, Query, Value; Vectorial Representations for every input.
#     * Query: The query is a representation of the current word used to score against all the other words (using their keys). We only care about the query of the token we’re currently processing.
#     * Key: Key vectors are like labels for all the words in the segment. They’re what we match against in our search for relevant words.
#     * Value: Value vectors are actual word representations, once we’ve scored how relevant each word is, these are the values we add up to represent the current word.
# 
# 7) *Score* : Dot product of Query Vector by each key vector followed by softmax.
#     * *Masked Score* : Dot product of Query Vector by first few key Vectors followed by softmax.
# 
# 8) *Attention Outcome* : We multiply each value by its score and sum up – resulting in our self-attention outcome.
# 
# 9) *Model Output* : When the top block in the model produces its output vector (the result of its own self-attention followed by its own neural network), the model multiplies that vector by the embedding matrix. Recall that each row in the embedding matrix corresponds to the embedding of a word in the model’s vocabulary. The result of this multiplication is interpreted as a score for each word in the model’s vocabulary.We can simply select the token with the highest score (top_k = 1). But better results are achieved if the model considers other words as well. So a better strategy is to sample a word from the entire list using the score as the probability of selecting that word (so words with a higher score have a higher chance of being selected). A middle ground is setting top_k to 40, and having the model consider the 40 words with the highest scores.
# 
# 
# 
# 
# 

# %% [markdown]
# Self Attention Mechanism : (https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
# 
# * Language heavily relies on Context.
# * the self-attention mechanism allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores.
# * The steps followed in self attention process :
#     1) Prepare inputs.
#     2) Initialise weights
#     3) Derive key, query and value
#     4) Calculate attention scores for Input 1
#     5) Calculate softmax
#     6) Multiply scores with values
#     7) Sum weighted values to get Output 1
#     8) Repeat steps 4–7 for Input 2 & Input 3
# 

# %% [markdown]
# 

# %%
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(torch.nn.Module):
    # A Neural Network layer implements Self Attention mechanism.
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value Projections.
        self.c_attn = torch.nn.Linear(config.n_embd, 3*config.n_embd) # It Ptojects the input Tensor into K, Q V.
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd) # It projects the output of self-attention into final output.
        # Dropout Regularization :
        self.attention_dropout = torch.nn.Dropout(config.dropout)
        self.residual_dropout = torch.nn.Dropout(config.dropout)
        # Masking :
        # [1 2 3 4]             [1 2 # #] ([1, 1, 0, 0]*[1 2 3 4])
        # [3  ]                   [3]
        self.register_buffer("bias", torch.trill(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head     # No.of attention heads in the self-attention.
        self.n_embd = config.m_embd  # The size of the input and output embeddings for the self-attention layer
    
    def forward(self, x):
        B, T, C = x.size() # B : Batch Size, T : Sequence Length, C : Embedding Dimensionality (n_embd)
        # K, Q, V computation : 
        q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# %% [markdown]
# Links:
# 
# 1) *Self Attention Mechanism* : https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
# 
# 2) *Python Dataclasses* : https://medium.com/mindorks/understanding-python-dataclasses-part-1-c3ccd4355c34
# 
# 3) *Pytorch Functional API* : 

# %%
class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4*config.n_embd)
        self.c_proj = torch.nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout  = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# %%
class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# %%
@dataclass
class GPTconfig:
    block_size : int = 1024 # Positional Encoding dimension
    vocab_size : int = 50257 
    n_layer :  int = 12
    n_head : int = 12
    n_embd : int = 768
    dropout : float = 0.1

class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe = torch.nn.Embedding(config.block_size, config.n_embd),
            drop = torch.nn.Dropout(config.dropout),
            h  = torch.nn.ModuleList([Block(config) for ele in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias = False)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def forward(self, idx, targets = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Forward the GPT Model :
        tok_emb = self.transformer.wte(idx) # Token_embeddings of shape = (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Desired Loss Calculation :
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = torch.nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        assert all(k=='dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # Override the dropout rate :
        if 'dropout' in override_args:
            config_args['dropout'] = config_args['dropout']

        config = GPTconfig(block_size=1024, **config_args)
        model = GPT(config)
        sd = model.state_dict()
        """
        In PyTorch, the learnable parameters (i.e. weights and biases) of a
        torch.nn.Module model are contained in the model’s parameters
        (accessed with model.parameters()). A state_dict is simply a
        Python dictionary object that maps each layer to its parameter tensor.
        """
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    # no_grad: a decorator that disables gradient calculation for the generate method. 
    # This is useful when the method is being called for generation purposes, rather than for training.
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond) # logits : (unnormalized log probabilities)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            	            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx







