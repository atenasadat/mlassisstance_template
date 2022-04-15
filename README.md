<div style="text-align: justify;">

<h1 align="center">
ML Project Boilerplate
</h1>

<div style="text-align: center;">

[![Deploy](https://github.com/AIMed-Team/ProjectBoilerplate/actions/workflows/deploy.yml/badge.svg)](https://github.com/AIMed-Team/ProjectBoilerplate/actions/workflows/deploy.yml)

</div>
    
This repository is a boilerplate template to start a project that uses the [`mlassistant`](https://github.com/AIMed-Team/mlassistant) framework.

## Usage Documentation

### Repository Initialization

1. Initialize your repository with this template repository.
1. Run the initialization command:
    
    ```bash
    ./init -p <project_name>
    ```

### Examples

1. There is an example of using the `mlassistant` framework to train a classifier model on the MNIST dataset. See the below files:
    - `config/mnist_config.py`
    - `data/mnist_loader.py`
    - `models/mnist_classifier.py`
    - `entrypoints/mnist.py`

### Start development

1. First, as the example shows, you need to implement your own `ContentLoader`(s), `Config`(s), `Model`(s), and `Entrypoints`(s). You may need to implement more classes by case.

1. This project structure is just a proposed one, and feel free to change it however you want. **But note that not to change the `entrypoints` package name because the entry points of your runs are read by `main.py` from this package.**


## Deployment

See [`mlassistant_deploy`](https://github.com/AIMed-Team/mlassistant_deploy#mlassistant-deploy).


1. Push the latest changes on the `master` branch.

1. Setup the server

    ```bash
    mlassistant-deploy setup [-D DEPLOY_PATH]
    ```

    Setup documentation is [here](https://github.com/AIMed-Team/mlassistant_deploy#setup).

1. Install `mlassistant`

    ```bash
    mlassistant-deploy install_mlassistant [-D DEPLOY_PATH] -V master
    ```

### Submit a run on queue

Create `submissions/<entrypoint>/<phase>.env` **config** file containing the following:

- `entrypoint` (*mandatory*): The name of your entrypoint file
- `phase` (*mandatory*): the The phase of your running
- `infra_conf` (*mandatory*): Configurations of the resources you need
- `queue` (*mandatory*): Name of the queue
- `run_args` (*optional*): By default, the command that will be run is the following:
    ```bash
    python main.py <entrypoint> <run_args>
    ```
    So, you can specify your additional args here. Do not use it if you don't want to use any extra arguments
- `command` (*optional*): If you do not want to run the above default command, set this variable to your desired command.

You can see an example of config file at `submissions/mnist/train.env`.
Submission documentation is [here](https://github.com/AIMed-Team/mlassistant_deploy#q-sub).

</div>
