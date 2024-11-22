
最近在复现 PPO 跑 MiniGrid，记录一下…


这里跑的环境是 Empty\-5x5 和 8x8，都是简单环境，主要验证 PPO 实现是否正确。


## 01 Proximal policy Optimization（PPO）


（参考：[知乎 \| Proximal Policy Optimization (PPO) 算法理解：从策略梯度开始](https://github.com) ）


首先，[策略梯度方法](https://github.com) 的梯度形式是


(1\)∇θJ(θ)≈1n∑i\=0n−1R(τi)∑t\=0T−1∇θlog⁡πθ(at\|st)然而，传统策略梯度方法容易一步走的太多，以至于越过了中间比较好的点（在参考知乎博客里称为 overshooting）。一个直观的想法是限制策略每次不要更新太多，比如去约束 新策略 旧策略之间的 KL 散度（公式是 plog(p/q)）：


(2\)DKL(πθ\|πθ\+Δθ)\=Es,aπθ(a\|s)log⁡πθ(a\|s)πθ\+Δθ(a\|s)≤ϵ我们把这个约束进行拉格朗日松弛，将它变成一个惩罚项：


(3\)Δθ∗\=arg⁡maxΔθJ(θ\+Δθ)−λ\[DKL(πθ\|πθ\+Δθ)−ϵ]然后再使用一些数学近似技巧，可以得到自然策略梯度（NPG）算法。


NPG 算法貌似还有种种问题，比如 KL 散度的约束太紧，导致每次更新后的策略性能没有提升。我们希望每次策略更新后都带来性能提升，因此计算 新策略 旧策略之间 预期回报的差异。这里采用计算 advantage 的方式：


(4\)J(πθ\+Δθ)\=J(πθ)\+Eτ∼πθ\+Δθ∑t\=0∞γtAπθ(st,at)其中优势函数（advantage）的定义是：


(5\)Aπθ(st,at)\=E(Qπθ(st,at)−Vπθ(st))在公式 (4\) 中，我们计算的 advantage 是在 新策略 的期望下的。但是，在新策略下蒙特卡洛采样（rollout）来算 advantage 期望太麻烦了，因此我们在原策略下 rollout，并进行 importance sampling，假装计算的是新策略下的 advantage。这个 advantage 被称为替代优势（surrogate advantage）：


(6\)Lπθ(πθ\+Δθ)\=J(πθ\+Δθ)−J(πθ)≈Es∼ρπθπθ\+Δθ(a∣s)πθ(a∣s)Aπθ(s,a)所产生的近似误差，貌似可以用两种策略之间最坏情况的 KL 散度表示：


(7\)J(πθ\+Δθ)−J(πθ)≥Lπθ(πθ\+Δθ)−CDKLmax(πθ\|\|πθ\+Δθ)其中 C 是一个常数。这貌似就是 TRPO 的单调改进定理，即，如果我们改进下限 RHS，我们也会将目标 LHS 改进至少相同的量。


基于 TRPO 算法，我们可以得到 PPO 算法。PPO Penalty 跟 TRPO 比较相近：


(8\)Δθ∗\=argmaxΔθ\[Lθ\+Δθ(θ\+Δθ)−β⋅DKL(πθ∥πθ\+Δθ)]其中，KL 散度惩罚的 β 是启发式确定的：PPO 会设置一个目标散度 δ，如果最终更新的散度超过目标散度的 1\.5 倍，则下一次迭代我们将加倍 β 来加重惩罚。相反，如果更新太小，我们将 β 减半，从而扩大信任域。


接下来是 PPO Clip，这貌似是目前最常用的 PPO。PPO Penalty 用 β 来惩罚策略变化，而 PPO Clip 与此不同，直接限制策略可以改变的范围。我们重新定义 surrogate advantage：


(9\)LπθCLIP(πθk)\=Eτ∼πθ\[∑t\=0Tmin(ρt(πθ,πθk)Atπθk,clip(ρt(πθ,πθk),1−ϵ,1\+ϵ)Atπθk)]其中， ρt 为重要性采样的 ratio：


(10\)ρt(θ)\=πθ(at∣st)πθk(at∣st)公式 (9\) 中，min 括号里的第一项是 ratio 和 advantage 相乘，代表新策略下的 advantage；min 括号里的第二项是对 ration 进行的 clip 与 advantage 的相乘。这个 min 貌似可以限制策略变化不要太大。


## 02 如何复现 PPO（参考 stable baselines3 和 clean RL）


* stable baselines3 的 PPO：[https://github.com/DLR\-RM/stable\-baselines3/blob/master/stable\_baselines3/ppo/ppo.py](https://github.com):[樱花宇宙官网](https://yzygzn.com)
* clean RL 的 PPO：[https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py](https://github.com)


代码主要结构如下，以 stable baselines3 为例：（仅保留主要结构，相当于伪代码，不保证正确性）



```


|  | import torch |
| --- | --- |
|  | import torch.nn.functional as F |
|  | import numpy as np |
|  |  |
|  | # 1. collect rollout |
|  | self.policy.eval() |
|  | rollout_buffer.reset() |
|  | while not done: |
|  | actions, values, log_probs = self.policy(self._last_obs) |
|  | new_obs, rewards, dones, infos = env.step(clipped_actions) |
|  | rollout_buffer.add( |
|  | self._last_obs, actions, rewards, |
|  | self._last_episode_starts, values, log_probs, |
|  | ) |
|  | self._last_obs = new_obs |
|  | self._last_episode_starts = dones |
|  |  |
|  | with torch.no_grad(): |
|  | # Compute value for the last timestep |
|  | values = self.policy.predict_values(obs_as_tensor(new_obs, self.device)) |
|  |  |
|  | rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones) |
|  |  |
|  |  |
|  | # 2. policy optimization |
|  | for rollout_data in self.rollout_buffer.get(self.batch_size): |
|  | actions = rollout_data.actions |
|  | values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions) |
|  | advantages = rollout_data.advantages |
|  | # Normalize advantage |
|  | if self.normalize_advantage and len(advantages) > 1: |
|  | advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) |
|  |  |
|  | # ratio between old and new policy, should be one at the first iteration |
|  | ratio = torch.exp(log_prob - rollout_data.old_log_prob) |
|  |  |
|  | # clipped surrogate loss |
|  | policy_loss_1 = advantages * ratio |
|  | policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range) |
|  | policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() |
|  |  |
|  | # Value loss using the TD(gae_lambda) target |
|  | value_loss = F.mse_loss(rollout_data.returns, values_pred) |
|  |  |
|  | # Entropy loss favor exploration |
|  | entropy_loss = -torch.mean(entropy) |
|  |  |
|  | loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss |
|  |  |
|  | # Optimization step |
|  | self.policy.optimizer.zero_grad() |
|  | loss.backward() |
|  | # Clip grad norm |
|  | torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) |
|  | self.policy.optimizer.step() |


```

大致流程：收集当前策略的 rollout → 计算 advantage → 策略优化。


计算 advantage 是由 rollout\_buffer.compute\_returns\_and\_advantage 函数实现的：



```


|  | rb = rollout_buffer |
| --- | --- |
|  | last_gae_lam = 0 |
|  | for step in reversed(range(buffer_size)): |
|  | if step == buffer_size - 1: |
|  | next_non_terminal = 1.0 - dones.astype(np.float32) |
|  | next_values = last_values |
|  | else: |
|  | next_non_terminal = 1.0 - rb.episode_starts[step + 1] |
|  | next_values = rb.values[step + 1] |
|  | delta = rb.rewards[step] + gamma * next_values * next_non_terminal - rb.values[step]  # (1) |
|  | last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam  # (2) |
|  | rb.advantages[step] = last_gae_lam |
|  | rb.returns = rb.advantages + rb.values |


```

其中，


* (1\) 行通过类似于 TD error 的形式（A \= r \+ γV(s') \- V(s)），计算当前 t 时刻的 advantage；
* (2\) 行则是把 t\+1 时刻的 advantage 乘 gamma 和 gae\_lambda 传递过来。


## 03 记录一些踩坑经历


1. PPO 在收集 rollout 的时候，要在分布里采样，而非采用 argmax 动作，否则没有 exploration。（PPO 在分布里采样 action，这样来保证探索，而非使用 epsilon greedy 等机制；听说 epsilon greedy 机制是 value\-based 方法用的）
2. 如果 policy 网络里有（比如说）batch norm，rollout 时应该把 policy 开 eval 模式，这样就不会出错。
3. （但是，不要加 batch norm，加 batch norm 性能就不好了。听说 RL 不能加 batch norm）
4. minigrid 简单环境，RNN 加不加貌似都可以（？）
5. 在算 entropy loss 的时候，要用真 entropy，从 Categorical 分布里得到的 entropy；不要用 \-logprob 近似的，不然会导致策略分布 熵变得很小 炸掉。


