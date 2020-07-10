# Game-of-Hog-Deep-Q-Learning-ANN
This project was created to learn more about reinforcement learning and how it works. The model didn't perform too well, with only a 59-60% win rate. With tuning,
the model did perform at 63-64%; however, the model was overly complex and unnecessary. Possible improvements might come from using a CNN instead of an ANN, changing
how rewards are given, changing the architecture of the current neural network, and more fine-tuning of hyperparameters. Feel free to improve if you want!

Ironically, upon further testing, it seems like basic Q-Learning actually performed better than the neural network model. The simple q-learning model performed at a 68-70% win rate which was quite shocking. I would assume that this is because the simple q-learning model essentially memorized the majority of states that occur in the game and acted accordingly. Possible improvements simply using this way of q-learning might come from tuning hyper parameters, changing reward distribution, and limiting our action space to more sensible moves.
