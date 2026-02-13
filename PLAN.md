# Plan: Fix Dataframe cell truncation

## Problem
When the user clicks a cell in the results table, Gradio expands it to show the full content (e.g., a 1000+ char protein sequence), taking up massive screen real estate. The desired behavior is:
- Cells visually truncated at column boundary (no wrapping, text clipped)
- Clicking a cell should NOT expand it into full-text edit mode
- Full text should be selectable/copyable (e.g., via the existing "Selected Match" detail pane on the right)

## Root cause
Two issues working together:
1. `interactive=True` on the Dataframe causes Gradio to enter "edit mode" on click, which expands the cell to show full content
2. No CSS `max-width` constraint on table cells, so even with `wrap=False`, cells can grow unbounded when clicked

## Plan

### Step 1: Set `interactive=False` on the Dataframe
- Change `gr.Dataframe(interactive=True)` → `gr.Dataframe(interactive=False)`
- This prevents the "expand on click" behavior entirely
- Users can still select text and copy with Ctrl+C
- The existing "Selected Match" detail pane (right column) already shows full sequences on row click

### Step 2: Remove Python-side truncation (the `…` ellipsis hack)
- Remove the `TRUNCATE` dict and the lambda that appends `\u2026`
- Send the **full data** to the Dataframe — let CSS handle visual truncation
- This means the clipboard will have the complete text when users select a cell

### Step 3: Add CSS for `text-overflow: ellipsis` on all cells
```css
#results-table table td {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    max-width: 250px;
}
```
- This visually clips text at the cell boundary with a proper ellipsis
- The underlying data is still complete (so copy works)

### Step 4: Set `column_widths` to give sensible proportions
- Give short columns (UniProt Entry, probabilities) small widths
- Give text-heavy columns (Query, Protein Name, Match Sequence) wider but capped widths

### Step 5: Verify the row-select handler still works
- The `results_table.select(fn=on_row_select, ...)` handler shows full details in the side panel
- With `interactive=False`, `.select()` should still fire on row click
- Verify this works — it's the main way users access full sequences

## Files to change
- `protein_conformal/backend/gradio_interface.py` (one file, three locations):
  1. CSS block (~line 1056)
  2. Truncation logic (~line 934-946) — remove
  3. Dataframe constructor (~line 1225) — set `interactive=False`, add `column_widths`
