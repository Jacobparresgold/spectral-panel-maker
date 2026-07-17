# Spectral Panel Picker

A small web app that picks the N most spectrally-distinct fluorophores
(fluorescent proteins + HaloTag/SNAPtag dyes) from a spectral-cytometer
unmixing-matrix export, balancing average pairwise separation against
worst-case (minimum) separation.

## Using it (no setup required)

Once deployed (see below), just open the app's URL. It ships with the lab's
default fluorophore library baked in (`data/`), so you can start picking
panels immediately -- no file needed. If you've measured new fluorophores
or want to use a different library, upload your own `.xlsx` unmixing-matrix
export in the sidebar; it should have the same layout as the CytExpert/
CytoFLEX export (a header row of fluorophore names, a "Channel" column, one
row per spectral channel).

## Running it yourself / editing it

```bash
git clone <this-repo-url>
cd spectral_panel_app
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

- `panel_selector.py` -- all the actual logic (data loading, the distance
  metric, the search algorithm, plotting). This is a plain Python module --
  import it directly if you'd rather script something than use the UI:

  ```python
  from panel_selector import load_spectra, classify_fluorophores, select_panel

  spectra, control = load_spectra("data/Unmixing_Matrix_20260715_unmixing_all.xlsx")
  fp_names, dye_names = classify_fluorophores(spectra)
  result = select_panel(spectra, N=8, M=1, fp_names=fp_names, dye_names=dye_names)
  print(result["panel"])
  ```

- `app.py` -- the Streamlit UI wrapping the above.
- `data/` -- the bundled default spectral library. Replace this file (keep
  the same name, or update `DEFAULT_DATA_PATH` in `app.py`) to change what
  loads by default for everyone.

## Deploying / updating the shared link for the lab

1. Push this folder to a GitHub repo (public, or private if you're using
   Streamlit Community Cloud's org/private-app tier -- see note on privacy
   below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   click "New app" -> point it at this repo, branch `main`, file `app.py`.
3. Streamlit builds and hosts it automatically; you'll get a URL like
   `https://<your-app-name>.streamlit.app` to share with the lab.
4. Any time you push a new commit (e.g. to `panel_selector.py`, `app.py`,
   or a new default `data/*.xlsx`), the deployed app redeploys automatically
   within a minute or two.

**Privacy note:** apps on Streamlit Community Cloud are visible to anyone
with the link by default (not indexed/searchable, but not access-controlled
either). If the spectral library or panel choices are sensitive, consider a
private GitHub repo + self-hosting (e.g. on institutional infrastructure)
instead.
