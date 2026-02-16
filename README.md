# Minifigures Webshop

## Welcome! 👋

Welcome to the *Minifigures Webshop* project assignment. The goal of this assignment is to teach the students everything they need to develop and deploy an AI solution from start to finish. For this project, you're going to learn how to create your own *Amazon.com* or *bol.com* like webshop, focussed on LEGO minifigures.

![Webshop Example](https://www.sendcloud.nl/wp-content/uploads//2018/04/Amazon-prime.png)

If you're curious how the final result will look like, go and visit our [reference solution](https://minifigures.radix.ai)!



## Getting Started 🚀

### Open in GitHub Codespaces

We recommend to develop in [GitHub Codespaces](https://github.com/features/codespaces), which will automatically open the pre-configured environment so that you can start coding immediately! Click on _Code_ and select _Create codespace_ to start a Dev Container GitHub Codespaces. By default, it will spin up a compute instance with a 2-core CPU, 8GB of RAM, and 32GB of storage. This is enough for this project and also the advised configuration since it prevents you from running out of core hours (240h/month for students).

### Develop in the devcontainer

By default, GitHub Codespaces will open a _Development Container_ (devcontainer) for you. Within this container, we've pre-configured a virtual environment for you, so you shouldn't worry about installing different packages and can start the project right away! Learn more about this virtual environment and its supported tools in the section "Development tools 🛠️" below.


## Development tools 🛠️

### Uv

This repository utilises on [uv](https://docs.astral.sh/uv/getting-started/features/), an environment that aims to make Python packaging and dependency management as easy as possible. To get you started, we've added all necessary requirements in the `pyproject.toml` file. However, in case you want to add other packages, you can do so using `uv add {package}` from within the development environment. Add `--group {dev|train|...}` to install the dependency as a training or development dependency, or create your own group. You can also remove packages from your environment using `uv remove {package}`.

### Poe the Poet

To make the development of your code go more efficiently, we've included a virtual assistant; [Poe the Poet](https://github.com/nat-n/poethepoet)! Run `poet` from within the development environment to see what it can do. A short summary of the available commands:

- `poe api` to run the REST API (found in the `src/minifigures_api/` folder)
- `poe app` to serve the Streamlit application (found in the `src/minifigures_app/` folder)
- `poe lint` to run the linting checks and fixes over your code
- `poe test` to test your code

