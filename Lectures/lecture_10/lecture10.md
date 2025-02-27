<div align="justify">

# Deep Q-Network (DQN)

> Goal: Approximate the optimal action-value function

<br>

$$
Q^*(s, a) = \max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \,\middle|\, s_0 = s, a_0 = a, \pi \right]
$$

<br>
Bellman Optimality Condition:

$$
Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} (s', a') \mid s, a \right]
$$

<br>

## Classical Q-Learning and the Bellman equation

Bellman Equation:
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

If 
$$
\alpha = 1 \rightarrow Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')
$$

<br>

## Approximation with a NN

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

<br>

## Stabilize learning: Two-Network Architecture

### Challenges
Instability due to non-stationary targets

<br>

### Policy Network
Estimate Q-values for current state

- Updated continuously at every training step

<br>

### Target Network
Provides stable target Q-values during the training of the Policy Network

- Copy of Policy Network but updated less frequently

<br>

### Implementation

- Action selection: Policy Network (state)

<br>

- Environmental interaction: Action executed $\rightarrow s', r$

<br>

- Experience storage: $(s, a, r, s')$ stored in replay buffer – Experience replay

<br>

- Sampling and Training

    - Sample from replay buffer – Avoids overfitting, improves generalisation

    - Compute target Q-value:
        $$
        y = r + \gamma \max\limits_{a'} Q_{\text{target}}(s', a')
        $$

<br>

- Updating the Policy Network

    $$
    L = \left[ y - Q_{\text{policy}}(s, a) \right]^2
    $$

    (Loss) $\rightarrow$ Update with gradient descent

<br>

q-value = policy_net(shape)

- dimension (batch_size, actions)

$$\begin{array}{c|cc}
 & A_1 & A_2 \\
\hline
B_1 & 24 & 12 \\
B_2 & 13 & 1 \\
\end{array}
$$

actions = [[0],[0]]

<br>

policy_net.gather(1, actions) $\Rightarrow$ [[24],[13]]

<br>

.squeeze() $\Rightarrow$ [24, 13] = q-value

target_net(next_state)

$$\begin{array}{c|cc}
 & a_1 & a_2 \\
\hline
b_1 & 2 & 0 \\
b_2 & 0 & 3 \\
\end{array}
$$

<br>

target_net.max(1) $\rightarrow$ ([2,3],[0,1])

[0] $\rightarrow$ [2,3] = next-q-value

target-q-value = r + $\gamma$ $\times$ next-q-value $\times$ (1-dones)

- If dones = True = 1

<br>

### Recall

Bellman Equation:
$$
Q(s,a) = r + \gamma \max\limits_{a'} Q(s', a')
$$


Where Q(s,a) is Policy_net and Q(s', a') is Target_net

<br>

$$
\Rightarrow \text{Loss} = \left| \text{target-q-value} - \text{q-value} \right|^2
$$

</div>

