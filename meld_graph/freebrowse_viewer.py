#!/usr/bin/env python3
"""
Generate a self-contained MRI viewer HTML file.
All data is embedded — no server required to view it.

Usage:
    python open_freebrowse.py --t1w T1w.nii.gz --flair FLAIR.nii.gz \
        --pial-lh lh.pial.T1 --white-lh lh.white --output viewer.html

Or edit the DEFAULT_* variables below and run:
    python open_freebrowse.py
"""

import argparse
import base64
import json
import sys
import tempfile
import urllib.request
import webbrowser
from pathlib import Path
from meld_graph.paths import FREEBROWSE_HTML

# FREEBROWSE_VERSION = "2.4.1"
# FREEBROWSE_HTML    = f"freebrowse-{FREEBROWSE_VERSION}.html"
# FREEBROWSE_DL_URL  = (
#     f"https://freesurfer.github.io/freebrowse/downloads/{FREEBROWSE_HTML}"
# )
# CACHE_DIR = Path(tempfile.gettempdir()) / "freebrowse_html_cache"


# def _download_freebrowse() -> Path:
#     CACHE_DIR.mkdir(parents=True, exist_ok=True)
#     dest = CACHE_DIR / FREEBROWSE_HTML
#     print(dest)
#     if not dest.exists():
#         print(f"Downloading FreeBrowse {FREEBROWSE_VERSION} (once) in {dest}…")
#         urllib.request.urlretrieve(FREEBROWSE_DL_URL, dest)
#         print("  Done.")
#     return dest

def _served_name(stem: str, src: Path) -> str:
    """Preserve source extension(s) so NiiVue infers the correct file format."""
    return stem + "".join(src.suffixes)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _generate_html(volumes: list, meshes: list) -> str:
    # freebrowse_html = _download_freebrowse().read_text(encoding="utf-8")
    freebrowse_html = Path(FREEBROWSE_HTML).read_text(encoding="utf-8")

    embedded = {}      # filename → base64
    nvd_volumes = []
    nvd_meshes  = []

    # parameters for volumes
    for stem, src, colormap, opacity, vmin, vmax in volumes:
        if src is None:
            continue
        name = _served_name(stem, src)
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"  Embedding {src.name}  ({size_mb:.1f} MB)…")
        embedded[name] = _b64(src)
        nvd_volumes.append({
            "url":      f"embedded://{name}",
            "name":     name,
            "colormap": colormap,
            "opacity":  opacity,
            "visible":  True,
            "cal_min": vmin,
            "cal_max": vmax,

        })
    # parameters for meshes (surfaces)
    for name, src, rgba in meshes:
        if src is None:
            continue
        size_kb = src.stat().st_size / 1024
        print(f"  Embedding {src.name}  ({size_kb:.0f} KB)…")
        embedded[name] = _b64(src)
        nvd_meshes.append({
            "url":             f"embedded://{name}",
            "name":            name,
            "rgba255":         rgba,
            "opacity":         1,
            "visible":         True,
            "meshShaderIndex": 14, # to make it crosscut
        })
    # general niivue parameters (e.g. radiological convention)
    opts = {"isRadiologicalConvention": True, "isNearestInterpolation": True, "sagittalNoseLeft":True}
    # combine all parameters
    scene = {"imageOptionsArray": nvd_volumes, "meshes": nvd_meshes, "opts": opts}
    scene['opts']
    scene_json = json.dumps(scene)
    embedded_json = json.dumps(embedded)

    # Injected before </body>: intercepts fetch() calls for embedded:// URLs
    # and for the NVD scene URL, returning in-memory data instead.
    inject = f"""
<script>
(function() {{
  const EMBEDDED = {embedded_json};
  const SCENE    = {scene_json};

  function b64ToResponse(b64) {{
    const raw = atob(b64);
    const buf = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
    return new Response(buf.buffer, {{
      status: 200,
      headers: {{'Content-Type': 'application/octet-stream'}},
    }});
  }}

  const _fetch = window.fetch.bind(window);
  window.fetch = function(input, init) {{
    const url = (input instanceof Request ? input.url : String(input));

    // Serve the scene NVD
    if (url.includes('scene.nvd')) {{
      return Promise.resolve(new Response(JSON.stringify(SCENE), {{
        status: 200,
        headers: {{'Content-Type': 'application/json'}},
      }}));
    }}

    // Serve embedded data files
    const name = url.replace('embedded://', '').split('/').pop().split('?')[0];
    if (EMBEDDED[name]) {{
      return Promise.resolve(b64ToResponse(EMBEDDED[name]));
    }}

    return _fetch(input, init);
  }};

  // Set the ?nvd= param so FreeBrowse loads the scene on startup
  if (!location.search.includes('nvd=')) {{
    const url = new URL(location.href);
    url.searchParams.set('nvd', 'embedded://scene.nvd');
    history.replaceState(null, '', url.toString());
  }}
}})();
</script>
"""
    # Insert the intercept script as early as possible so fetch is patched
    # before FreeBrowse's own JS runs.
    patched = freebrowse_html.replace("<head>", "<head>" + inject, 1)
    if patched == freebrowse_html:
        patched = inject + freebrowse_html
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Generate a self-contained MRI viewer HTML (no server needed)."
    )
    parser.add_argument("--t1w",      default='T1w.nii.gz',      help="T1w volume (.nii.gz or .mgz)")
    parser.add_argument("--flair",    default=None,    help="FLAIR volume (.nii.gz or .mgz)")
    parser.add_argument("--pial-lh",  default=None,  help="LH pial surface")
    parser.add_argument("--white-lh", default=None, help="LH white surface")
    parser.add_argument("--pial-rh",  default=None,  help="RH pial surface")
    parser.add_argument("--white-rh", default=None, help="RH white surface")
    parser.add_argument("--lesion",   default=None,             help="Lesion mask overlay (.nii.gz or .mgz)")
    parser.add_argument("--output",   default='viewer.html',   help=f"Output HTML file (default: viewer.html)")
    args = parser.parse_args()

    def _resolve(val):
        return Path(val) if val else None

    t1w      = _resolve(args.t1w)
    flair    = _resolve(args.flair)
    pial_lh  = _resolve(args.pial_lh)
    white_lh = _resolve(args.white_lh)
    pial_rh  = _resolve(args.pial_rh)
    white_rh = _resolve(args.white_rh)
    lesion   = _resolve(args.lesion)

    named = {
        "--t1w": t1w, "--flair": flair,
        "--pial-lh": pial_lh, "--white-lh": white_lh,
        "--pial-rh": pial_rh, "--white-rh": white_rh,
        "--lesion": lesion,
    }

    missing = [f"{flag}: {p}" for flag, p in named.items()
               if p is not None and not p.exists()]
    if missing:
        print("Error — file(s) not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    if all(p is None for p in named.values()):
        print("Error — no files provided. Use --t1w, --flair, --pial-lh, etc.")
        sys.exit(1)

    # tell name, path, colormap, opacity, vmin, vmax
    volumes = [
        ("FLAIR",  flair,  "gray",   1.0, None, None),
        ("T1w",    t1w,    "gray",   1.0, None, None),
        ("lesion", lesion, "random", 0.8, 0, 1),
    ]
    meshes = [
        ("lh.pial",  pial_lh,  [255, 255, 0, 255]),
        ("lh.white", white_lh, [255, 165, 0, 255]),
        ("rh.pial",  pial_rh,  [255, 255, 0, 255]),
        ("rh.white", white_rh, [255, 165, 0, 255]),
    ]

    output = Path(args.output)
    print(f"Generating {output}…")
    html = _generate_html(volumes, meshes)
    output.write_text(html, encoding="utf-8")
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Done — {output}  ({size_mb:.1f} MB)")
    webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
