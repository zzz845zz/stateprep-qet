# Intall Poetry

```bash
pip install pipx
pipx install poetry
```

# Setup

```bash
poetry env use {python3, python}    # create a virtual environment
poetry install                      # install dependencies
```

# Run python file

```bash
poetry run {python3, python} {file.py}
```

Example:

```bash
poetry run python ./tests/classiq_qsvt.py
poetry run python ./tests/pennylane_qsvt.py
```