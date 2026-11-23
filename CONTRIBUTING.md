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

**Code graph (optional)**

`graphify-out/graph.json` is a [graphify](https://graphify.com) map of the code
that AI assistants query instead of grepping. `prek` rebuilds it on every
commit; if it changed, `git add graphify-out/graph.json` and commit
again. Register the merge driver once per clone so two branches that both
changed the graph merge without a conflict:

```bash
uv tool install graphifyy
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$(sed -n '1s/^#!//p' "$(command -v graphify)")\" -m graphify merge-driver %O %A %B"
```

If a merge still conflicts in the graph, rebuild it: `rm graphify-out/graph.json && graphify update . && git add graphify-out/graph.json`.

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

If a merge, rebase, or cherry-pick conflicts in `uv.lock`, don't resolve it by
hand or by picking a side; regenerate it against the merged `pyproject.toml`:

```bash
git checkout --ours uv.lock && uv lock && git add uv.lock
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
uv run jupyter nbconvert --to markdown docs/examples/*.ipynb --output-dir docs/examples/
uv run zensical serve
```

**Using pip:**

```bash
pip install -e . --dependency-groups docs
jupyter nbconvert --to markdown docs/examples/*.ipynb --output-dir docs/examples/
zensical serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.

Note: the `jupyter nbconvert` step converts the tutorial notebooks to Markdown before serving. Re-run it whenever you edit a notebook.
