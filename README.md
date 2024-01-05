# CNN_Project using ML OPS

### Useful commands:
For building a running a docker container on cpu linux (using WSL) you can do this:
Remember 
```
docker build -f trainer.dockerfile . -t trainer:latest
```

For running a docker container we use:
Remember using the -v command we save the output of the models folder to our machine
```
docker run --name 5715fbb43247 -v ./models:/models/ trainer:latest
```

It is also possible to use the "make commandos" like
```
make data
make train
```

 and so on


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
