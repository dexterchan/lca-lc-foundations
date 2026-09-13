---
name: notebook-to-vscode-py
description: Converts a Jupyter notebook (.ipynb) into a plain .py file using VSCode's "# %%" cell-marker format, so it can be run block-by-block in VSCode's Python Interactive window instead of the Jupyter browser UI. Use this whenever the user wants a notebook turned into a script, wants to "get out of Jupyter", mentions VSCode cell markers / "# %%" / Python Interactive, or wants to step-debug notebook code with a real debugger.
---

# Convert a notebook to a VSCode-runnable .py file

## Why this format

VSCode recognizes a line starting with `# %%` as a cell boundary. Each cell gets a "Run Cell"
codelens above it, and `Shift+Enter` runs the current cell and moves to the next — same muscle
memory as Jupyter, but backed by VSCode's real debugger (breakpoints, step-in, watch variables)
instead of the browser notebook UI.

## Step 1: Read the notebook

Read the `.ipynb` file. Each cell becomes one `# %%` block in the output, in the same order.

## Step 2: Map each cell type

- **Code cell** → `# %%` on its own line, then the cell's source verbatim on the following lines.
- **Markdown cell** → `# %% [markdown]` on its own line, then each line of the markdown content
  prefixed with `# ` on the following lines (so it stays a valid Python comment but VSCode still
  renders it as a markdown cell).
- **Empty trailing cell** → skip it, don't emit a bare `# %%` with nothing under it.

Leave a single blank line between the `# %%` marker and the cell content, and a blank line after
each cell's content before the next `# %%` — matches VSCode's own notebook-to-script export format.

## Step 3: Don't change the code

This is a format conversion, not a rewrite. Keep every line of code exactly as it is in the
notebook — same imports, same variable names, same logic. If the user separately wants the code
itself changed (e.g. converted to OpenRouter), that's a different task — ask, don't bundle it in
silently.

## Step 4: Where to save it

Save as `<notebook-name>.py` next to the original `.ipynb` (same directory, same base name, `.py`
extension). Don't delete or modify the original `.ipynb` unless the user asks — some people want
to keep both.

## Step 5: Tell the user how to use it

After creating the file, remind them: open it in VSCode, use "Run Cell" above a block or put the
cursor in a block and hit `Shift+Enter`. The first run of any block will prompt VSCode to start a
Python Interactive kernel using the project's environment.
