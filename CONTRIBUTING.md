# Contributing

## Making a pull request

Fork the repo on GitHub.

```bash
# clone fork
git clone https://github.com/USER/tinyspace.git

# add original repo
git remote add upstream https://github.com/etaoxing/tinyspace.git

# keep fork up to date
git checkout master
git pull --rebase upstream master
git push
```

Make code changes in a branch.
```bash
# create feature branch
git checkout -b BRANCH
git push -u origin BRANCH

# setup dev environment
conda create -n PACKAGE python=3.7
conda activate PACKAGE
pip install -e ".[all]"

# pre-commit hooks
pre-commit install
```

Run linting and tests.
```bash
isort .
black .
flake8 .
pytest -rvP --forked --cov=tinyspace tests/
```

Document your code, following a [looser version](https://drake.mit.edu/styleguide/pyguide.html) of Google style docstrings.

Update the [`CHANGELOG`](./CHANGELOG.md), in the "Unreleased" section.

Open a new pull request on GitHub.

## Building documentation

```bash
cd docs/
make html # or `make livehtml` for autobuild
```

## Releasing

