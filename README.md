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

Next things to integrate:
* evaluation
* load model instead of create new one

To run experiments on the cluster:
* The dockerfile needs the requirements file when image is build
* The script file (final CMD command in docker file) needs to be created and fed to the container per volume

