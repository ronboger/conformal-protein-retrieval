# Handoff: Gradio Table Cell Clipping + Click-to-Expand

## The Goal

The user wants a **Google Sheets-like** table experience:
1. Long text in cells is **visually clipped** (not truncated with "...") — the full data is in the cell, but overflow is hidden
2. Clicking a cell **expands it in-place within the table** to reveal the full content
3. Clicking again (or clicking elsewhere) collapses it back

This should apply to all potentially long columns: `query_meta`, `lookup_seq`, `lookup_meta`, `lookup_protein_names`.

## Current State

The table currently uses **explicit string truncation** (adding "..." to the data) as a workaround. This makes the table readable but:
- The cell data is modified (not just visually clipped)
- Clicking a cell in edit mode shows the TRUNCATED value, not the full value
- Full content is only accessible via the detail panel on the right (click a row → `gr.Code` panel shows full sequences/metadata)

The user finds this acceptable but wants the Google Sheets behavior instead.

## What Was Tried

### 1. `wrap=False` on `gr.Dataframe` (partially worked)
- **What it does**: Sets `white-space: nowrap` on cells
- **Result**: Prevents line wrapping but does NOT constrain column width. Columns expand horizontally to fit content, so long sequences make the table very wide with a horizontal scrollbar. No visual clipping.
- **With `interactive=True`**: Clicking a cell enters edit mode, showing an input field with full text. This IS the expand behavior the user wants — but only works if the initial view is clipped.

### 2. CSS `table-layout: fixed` + `overflow: hidden` on `#results-table`
```css
#results-table table { table-layout: fixed; width: 100%; }
#results-table table th, #results-table table td {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 0;
}
```
- **Result**: No visible effect. Gradio 5's Dataframe component likely uses scoped styles or a rendering approach that overrides external CSS. The CSS selectors may not match the actual DOM structure.
- **Note**: Gradio 5 uses Svelte components. The Dataframe may render with scoped class names (e.g., `.svelte-xxxxx`) that have higher specificity than `#results-table` selectors.

### 3. Explicit string truncation with "..." (current approach)
```python
TRUNCATE = {"query_meta": 50, "lookup_seq": 40, ...}
display_df[col] = display_df[col].apply(lambda v: str(v)[:lim] + "…" if len(v) > lim else v)
```
- **Result**: Table is readable. But clicking a cell shows truncated text, not full text.
- **Detail panel**: Full content accessible by clicking a row → `gr.Code` panel on the right.

### 4. CSS `max-width` on `#results-table td` (did not work)
- Similar to approach 2. Gradio's internal styles override.

## Approaches NOT Yet Tried

### A. JavaScript via `gr.Blocks(js=...)` (most promising)
Use inline styles applied by JS (highest CSS specificity — can't be overridden):
```javascript
() => {
    const applyClipping = () => {
        const table = document.querySelector('#results-table table');
        if (!table) return;
        table.style.tableLayout = 'fixed';
        table.style.width = '100%';
        for (const cell of table.querySelectorAll('td, th')) {
            cell.style.overflow = 'hidden';
            cell.style.textOverflow = 'ellipsis';
            cell.style.whiteSpace = 'nowrap';
            cell.style.cursor = 'pointer';
        }
    };
    new MutationObserver(applyClipping).observe(document.body, {childList: true, subtree: true});

    document.addEventListener('click', (e) => {
        const td = e.target.closest('#results-table td');
        if (!td) return;
        const expanded = td.dataset.expanded === 'true';
        td.style.whiteSpace = expanded ? 'nowrap' : 'normal';
        td.style.overflow = expanded ? 'hidden' : 'visible';
        td.style.textOverflow = expanded ? 'ellipsis' : 'clip';
        td.dataset.expanded = expanded ? 'false' : 'true';
    });
}
```
- **Why this might work**: Inline styles have the highest specificity. MutationObserver reapplies after Gradio re-renders.
- **Risk**: The `#results-table table td` selector might not match Gradio 5's DOM. Need to inspect the actual rendered DOM.

### B. Inspect the actual Gradio 5 DOM
Use browser DevTools on the deployed app to find:
1. What HTML elements Gradio renders for table cells
2. What CSS classes/styles are applied
3. What selector would actually match

Then write targeted CSS or JS.

### C. `gr.Dataframe(column_widths=[...])` parameter
Gradio 5's Dataframe may support `column_widths` to set fixed widths per column. If columns have fixed widths and `wrap=False` is set, content should clip.
- **Issue**: Number of columns varies by result, and `column_widths` is set at component creation time.
- **Possible fix**: Return `gr.Dataframe(value=df, column_widths=[...])` from handler functions.

### D. Replace `gr.Dataframe` with `gr.HTML`
Render the table as raw HTML with full CSS control. Loses Gradio's interactive features (sorting, editing, selection) but gains complete control over styling.

### E. Use `gr.Dataframe(datatype=...)` with explicit column types
Some Gradio versions support per-column settings that may include max width.

## Key Files

| File | What to change |
|------|---------------|
| `protein_conformal/backend/gradio_interface.py` | Lines ~933-947 (main display), ~1502-1517 (filter_by_query display), ~1076 (`gr.Blocks` creation), ~1207-1212 (`gr.Dataframe` creation) |

## Important Context

- **Deployment**: `modal deploy modal_app.py` — deploys to Modal cloud
- **Gradio version**: `>=5.0.0` (installed in Modal container, see `modal_app.py` line 51)
- **The Dataframe has `elem_id="results-table"`** — use this for CSS/JS targeting
- **The Dataframe has `interactive=True`** — clicking cells enters edit mode (this IS the expand mechanism if clipping works)
- **Full data is in `CURRENT_SESSION["results"]["matches"]`** — the detail panel (`sequence_detail`) reads from here on row click
- **Tests**: `pytest tests/ -v` (51 tests, run on compute nodes via SLURM)

## Verification

After fixing, confirm:
1. Table cells show clipped content (no horizontal scrollbar, no "..." in data)
2. Clicking a cell expands to show full content within the table
3. Clicking away / clicking again collapses it
4. The detail panel still works (click a row → full sequences in right panel)
5. Query filter dropdown still works
6. Probability plot still renders
