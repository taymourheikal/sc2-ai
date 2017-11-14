Version 0 notes and improvements to be made:

This version took a couple of days to implement once the PySC2 documentation and code was (somewhat) sufficiently understood.

This is only a small modification on Steven Brown's tutorial code on building a smart agent (reward structure is different, and included full list of actions the agent can take at any timestep in the quest for full autonomy). The tutorial was very helpful, and his code is available on GitHub as well; I highly recommend you read it if you've installed PySC2 and are asking yourself "now what?": https://chatbotslife.com/building-a-smart-pysc2-agent-cdc269cb095d

Notes & Next Improvements:
- This version does not include a Deep Q Network (DQN); strictly built as a simple Q-Learning agent. This means that performance can be somewhat stable at the beginning of a game, but it quickly tends towards complete randomness.
- States and rewards are very limited and need to be expanded. The state is passed as a numeric list 5 elements long, including supply count, mineral collection rate, and minerals spent. There are 8 possible rewards + negative rewards. The possibility of incorporating a Convolutional Neural Network is being investigated.
- The agent has no sense of space whatsoever. Once the agent decides on an action (e.g. build a supply depot), the current version completely randomizes the x and y coordinates on which the action is to be done. To solve this, some sort of layered decision-making process will have to be implemented.
- Extracting the agent's build order over epochs can prove to be a useful in tool in tracking the agent's behavior changes.

For v1, the main goal is to redesign the agent's brain into a DQN (1), and at least pass forward a more comprehensive vector of what the current state is (2). Additionally, recoding the reward system from a series of 'if' statements to something more efficient (like vectors or matrices) will prove to be helpful in later versions.

Wins:
- On most epochs, the agent decides to leave the SCVs mining from the minerals, instead of immediately sending them out for random actions. I hope this is due to the negative reward on a reduction in mineral rate collection.
- The agent is rewarded based on minerals spent, and so occassionally will build SCVs, and will often initiate building a structure, although has no regard for whether or not the building is actually completed.
