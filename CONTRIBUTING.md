# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

**Getting started**

First fork the library on GitHub.

Then clone and install the library:

```bash
git clone https://github.com/your-username-here/quax.git
cd quax
```

**Using uv:**

```bash
uv sync --group dev
prek install  # `prek` is installed by `uv` on the previous line
```

**Using pip:**

```bash
pip install -e . --dependency-groups dev
prek install  # `prek` is installed by `pip` on the previous line
```

---

**If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

Next verify the tests all pass:

**Using uv:**

```bash
uv run pytest
```

**Using pip:**

```bash
pip install -e . --dependency-groups tests
pytest
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

**Using uv:**

```bash
uv run mkdocs serve
```

**Using pip:**

```bash
pip install -e . --dependency-groups docs
jupyter nbconvert --to markdown docs/examples/*.ipynb --output-dir docs/examples/
zensical serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.

Note: the `jupyter nbconvert` step converts the tutorial notebooks to Markdown before serving. Re-run it whenever you edit a notebook.
