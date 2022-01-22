# Prompt-Tuning-for-NLG
Repository for Prompt Tuning for Natural Language Generation tasks.

An example in how to design possible experiments can be seen in the follwoing end-to-end test case:
* src > test > end-to-end > test_simple_soft_prompts_inference

This test case does not do any training nor loads it a model. It uses the AMR dataset.


To call the test runner and run all test cases run the following code snippet in this projects main folder:
```
python ./src/test/test_runner.py
```

To run a specific test use module "unittest" and the -v parameter with the name of the test file :
```
python -m unittest -v src/test/end_to_end/test_simple_soft_prompts_inference.py
```

Next things to integrate:
* reading arguments from command line 
  * (I already have code snippets for it in another branch > will integrate it later)
* reading experiments via yaml config files
  * (I already have code snippets for it in another branch > will integrate it later)

To run experiments on the cluster:
* If created a docker file > inspired by the one we used in another seminar last semester > it is based on CUDA docker file
  * The dockerfile needs the requirements file when image is build
  * The script file (final CMD command in docker file) needs to be created and fed to the container per volume
* Haven't tested it yet > I plan to do tonight
