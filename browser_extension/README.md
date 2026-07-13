# Fake-News Screening — browser extension (skeleton)

A minimal Manifest V3 extension that sends text to the screening API
(`src/api.py`) and shows the verdict. It is a **scaffold** for the productization
path, not a published extension.

## Run it

1. Start the API (from the project root):

   ```bash
   pip install -r requirements-api.txt
   uvicorn src.api:app --reload      # serves http://127.0.0.1:8000
   ```

2. Load the extension:
   - Chrome/Edge → `chrome://extensions` → enable **Developer mode** →
     **Load unpacked** → select this `browser_extension/` folder.

3. Use it:
   - Click the toolbar icon, paste text, **Screen text**; or
   - select text on any page → right-click → **Screen “…” for fake news**.

## Notes / to finish before publishing

- Add an `icon128.png` (referenced by the notification and recommended in the
  manifest `action`). Any 128×128 PNG works.
- `API_URL` in `popup.js` / `background.js` and `host_permissions` in
  `manifest.json` point at `http://127.0.0.1:8000`. Change all three to your
  deployed API origin, and tighten `API_ALLOWED_ORIGINS` (see `src/config.py`)
  to your extension id / domain.
- The API contract is the stable `/screen` response shaped by
  `src.api.build_response` (unit-tested in `tests/test_api.py`).
