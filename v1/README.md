**Version 1 notes:**

The biggest change in this version is the introduction of a Deep Q Network. The code in dqn.py was adapted from **Morvan Zhou's tutorial** on Reinforcement Learning: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow . Very helpful and highly recommended if your new to RL (as is my case :) ). The other change implemented is a redesign of the reward structure from a series of if-else statements to np.array operations. This will have benefits later on when a more complex reward system is introduced.

I'll upload a video to YT for this version when the below improvements have been made/attempted (v1.1).

**Notes and Next Improvements:**
- Introduce some mechanism that allows the agent to make hierarchical decisions. For example, once the the action is chosen (build depot), where should the agent execute the action. At the moment, it's completely random; that's problematic because it could get punished for seemlingly good actions, like building a supply depot [good] in the enemy base [bad].
- Introduce minimap and screen feature layers as an input. This should help with the above issue, since you usually want to build structures on high terrain, and next to existing structures and allied units.
- Try out a double DQN, which according to van Hasselt, Guez, and Silver, fixes the issue with DQNs tending to overestimate action values: https://arxiv.org/pdf/1509.06461.pdf .

**Wins:**
- I haven't done much to test the difference in performance between v0 and v1, but through just watching it in action, the DQN does seem to make the agent more stable throughout the episode. It's also much better at spending it's minerals. For some reason it has a fascination with engineering bays and turrets...
- Hyperparameters have not been tuned, so given the aforementioned jump in performance, I'm hoping that tuning the hyperparameters and trying out different NN functions can also make a big difference. I am planning to do some comparisons when I tune some of the hyperparameters.
