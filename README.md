# Prompt-Tuning-for-NLG
Repository for Prompt Tuning for Natural Language Generation tasks.

To build the docker image run the following code snippet in this projects main folder:
```
docker image build -t midl_nlg:test .
```

To run a container run the following command:
```
docker container run --rm --volume="$(pwd):/opt/workspace" midl_nlg:test
```

You can built an enroot image and run containers via enroot:
```
enroot create --name <midl_nlg_test> <name_of_your_eonroot_image.sqsh>
enroot list -f
enroot start --rw --mount .:/opt/workspace <midl_nlg_test>
```

Or manually run an experiment via python:
```
python main.py --experiment <name_of_experiment.yml>
```
The experiment-yaml-file is expected to be found in the experiments folder of this project.