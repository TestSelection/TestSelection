# Test Data Selection based on Uncertainty Matrix

- Why ?
  - Deep Learning is a kind of statistical learning. To test the the deep learning model,
  we need stand at an abstract point. Assuming the true data generation follows data generation `G`, 
  you want to know how far you model can fit the true generation distribution ( `accuracy` ), and how stable your model 
  can fit the true generation in the different scnerios (`uncertainty` ). We regard the model setting and trainning strategies as a `black box`. We use
  the uncertainty and covrage based on the outputs of model to test Deep Learning. The project studies these diffrent test metrics for this problem. We use them
  to socre data, select them, and show how to use the results to improve Deep Learning. Befotre testing, you should make sure that the output of your model currently have the similar distribution with the training dataset you use, e.g., good accuracy or low loss. Otherwise, it makes no much sense to test a bad performance model.

- Implementation
  - python 3.6
  - foolbox, adversarial attacking
  - tensorflow 1.8
  - Keras
  - We resue the code from Surprised Adequacy (https://arxiv.org/pdf/1808.08444.pdf) and DeepTest.  
  

