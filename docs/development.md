# Development

## Tests

The tests follow the behaviour described in these docs. Most of them use simple causality functions; the tests in
`tests/test_pc_causality.py` use the real PC of gCastle and are skipped when gCastle (and torch) cannot be imported.

```bash
pip install -e ".[test,pc]"
python -m pytest
```

On Windows, create the virtual environment in a short path (e.g. `C:\venvs\pdmc`): torch has deep folders and the
installation fails with "The filename or extension is too long" in long paths.

## Documentation

The documentation is built with MkDocs (Material theme) and mkdocstrings, from `docs/` and the docstrings:

```bash
pip install -r docs/requirements.txt
mkdocs serve            # preview at http://127.0.0.1:8000
mkdocs build --strict   # static site in site/
```

The notice at the top of the home page is included from `CHANGELOG.md` (between the `notice` markers).
