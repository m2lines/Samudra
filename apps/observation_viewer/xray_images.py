# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Client-side pending feedback for saved Xray image updates."""

import panel as pn
import param


class XrayImages(pn.reactive.ReactiveHTML):
    content = param.Dict(default={"html": ""})
    busy = param.Boolean(default=False)
    selection = param.String(default="")
    _template = """
    <div style="position:relative;width:100%">
      <div id="images"></div>
      <div id="status" role="status" aria-live="polite"
           style="display:none;position:absolute;right:8px;top:8px;padding:3px 7px;
                  background:rgba(255,255,255,.94);border:1px solid #aaa;border-radius:4px;
                  font-size:12px;color:#173b4a;pointer-events:none;z-index:5">Loading…</div>
    </div>
    """
    _scripts = {
        "render": "self.content()",
        "busy": """
          status.style.display = data.busy ? 'block' : 'none';
          if (data.busy) state.version = (state.version || 0) + 1;
        """,
        "content": """
          const version = (state.version || 0) + 1;
          state.version = version;
          images.innerHTML = data.content.html;
          const pending = Array.from(images.querySelectorAll('img')).map(img =>
            img.decode ? img.decode().catch(() => {}) : Promise.resolve());
          Promise.all(pending).then(() => requestAnimationFrame(() => {
            if (state.version === version && (!data.selection ||
                data.selection === JSON.stringify(data.content.selection))) data.busy = false;
          }));
        """,
    }
