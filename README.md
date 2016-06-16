# Deep Q-Networks for Accelerating the Training of Deep Neural Networks

> Source code to the paper [https://arxiv.org/abs/1606.01467](https://arxiv.org/abs/1606.01467)

> My current thoughts on hyperparameter optimization can be found in my [blog article](https://bigaidream.gitbooks.io/tech-blog/content/2016/thoughts-hyperparameter.html).

## Casual abstract and our motivations
![](https://github.com/bigaidream-projects/qan/blob/master/angry_catapult.jpg)

This paper is partially inspired by Angry Birds.

 All the other attempts ignore the side-effect of the global update machanism of back-propagation. We add a regressor to every layer to make sure that the positions of neurons do not change at every episode. It does not make sense for the Atari games' to change their graphics APIs at every episode. We suspect that this is the main reason other methods need hundreds of thousands of episodes, whereas we only need 20 episodes on MNIST dataset. 

 Then we realize that the regressor itself can actually itself accelerate the overall hyperparameter tuning process significantly. Obviously, if we only gradually change some hyperparameters, the training trajectories of the DNN being tuned by DQNs should not differ significantly between episodes. We can think of the regressor as momentum used in stochastic gradient descent algorithms. Oh, this is also like the catapult used in Angry Birds...

## Reproduce our results on MNIST

### Dependencies
We are using Lua/Torch. The DQN component is mostly modified from [DeepMind Atari DQN](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner). 

You might need to run `install_dependencies.sh` first. 

### Tuning learning rates on MNIST
```bash
cd mnist_lr/;
cd mnist;
th train-on-mnist.lua; #get regression filter, save in ../save/
./run_gpu; #Start tune learning rate using dqn
#To get the test curve, run following command
cd mnist_lr/dqn/logs;
python paint_lr_episode.py;
python paint_lr_vs.py;
```

### Tuning mini-batch selection on MNIST 
```bash
cd mnist_minibatch;
cd mnist;
th train-on-mnist.lua; #get regression filter, save in ../save/
./run_gpu; #Start select mini-batch using dqn
#To get the test curve, run following command
cd mnist_minibatch/dqn/logs;
python paint_mini_episode.py;
python paint_mini_vs.py;
```

### Different Settings
1. GPU device can be set in `run_gpu` where `gpu=0`
2. Learning rate can be set in `/ataricifar/dqn/cnnGameEnv.lua`, in the `step` function. 
3. When to stop doing regression is in `/ataricifar/dqn/cnnGameEnv/lua`, in line 250

---

## FAQ
1. `Q:` Do the actions change over time? Do they converge to something like the alternated classes with uniform sampling heuristics that is always used in CNNs? 
`A:` Yes, from what we observed, the actions do change over time. We are making a visualization tool to show what actions have been taken at every iteration. 

2. `Q:` Is the regression really necessary?
`A:` We think the added regression operation is the most important factor making our QAN converge so faster. Actually this is also extremely useful for all hyperparameter optimization tasks. Thus we already integrated this trick into another hyperparameter tuning tool [DrMAD](https://github.com/nicholas-leonard/drmad). But we are still investigating exactly to what extent this regressor contribute to the overall performance. 


## Citation
```
@article{dqn-accelerate-dnn,
  title={Deep Q-Networks for Accelerating the Training of Deep Neural Networks},
  author={Fu, Jie and Lin, Zichuan and Liu, Miao and Leonard, Nicholas and Feng, Jiashi and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1606.01467},
  year={2016}
}
```

## Contact

If you have any problems or suggestions, please contact: jie.fu A~_~T u.nus.edu~~cation~~