# 2048-RL
An approach to solving the 2048 game using reinforcement learning in the context of a coding challenge

About the challenge
---

The challenge is to implement the game logic and provide AI hints to the player if requested. Since adding some UI
on top of this game is quite straight forward, I decided to go for a simple command line interface. The actual challenge
is to provide meaningful AI hints for the player. A LLM is for sure the wrong tool for such a task as it does not 
require interpreting written instructions or to formulate out answers. Of course, an LLM could help with providing 
"human-readable" explanations for the suggestion, but that's an even more complex task to solve. The value of the AI is 
in providing one of four directions as "best next move". Studying the game shows that this is not a simple task as 
seemingly harmless moves can be devastating down the road. However, the game mechanics are so simple that I suspect a 
small decision tree would perform better and much more resource efficient than a more complex AI model. But that's not 
the challenge.

As for the AI model: Looks like a nice task to try out some reinforcement learning, but I do not have any prior 
experience in this field. I give it a go anyway, I am curious what I can learn.


Game engine
---

For the game engines implementation just the python standard library is used. Since the game logic is straightforward, 
and the state is small, this will be reasonably fast when running RL on it where I would expect the majority of time 
will be spent updating the model. This probably could have been solved much more elegantly using numpy or similar, but 
I wanted to keep it simple and therefore did not introduce additional dependencies for this (yet).

