# grachev.ai

Denis Grachev’s personal homepage. Plain HTML, CSS and JavaScript modules. No build step: everything the pages need is checked in.

## Preview

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. The robot needs to be served over HTTP; opening `index.html` from disk will not load it.

## Layout

- `index.html`, `style.css` — the homepage. To add a paper, copy an `<article class="entry">` block in the Research section.
- `playground/` — the robot.
  - `sim.js` — MuJoCo physics, inverse kinematics, the grab/put-down logic and the self-play. No rendering.
  - `view.js` — three.js drawing and mouse/touch input.
  - `stage.js` — lazy loading and the status line; used by both pages.
  - `model/` — a Franka Panda arm with a Robotiq 2F-85 gripper, both from MuJoCo Menagerie (licences in `model/LICENSE.*`). The teal strips on the fingertips stand in for tactile sensors: the page reads MuJoCo contact forces on the pads and draws them as a small grid. Meshes are decimated and packed into `meshes.json` by `tools/pack_meshes.py`. `scene.xml` holds the toys.
  - `index.html` — the full-size playground. Its “real one” section embeds the hosted Rerun viewer (`app.rerun.io`) on click, pointed at `recordings/stack.rrd`. `data-version` on `#rerun` must match the Rerun version that wrote the file. The hosted viewer cannot fetch from localhost, so on localhost the page uses a local viewer instead: run `rerun --serve-web --web-viewer-port 9391` and serve the site with CORS headers (`Access-Control-Allow-Origin: *`).
  - `recordings/` — `stack.rrd`, a real-robot episode shrunk for the web, and its poster image.
- `vendor/` — `@mujoco/mujoco` 4.0.1 (single-threaded WASM build) and three.js 0.170.0.
- `tools/pack_meshes.py` — rebuilds `meshes.json` from a MuJoCo Menagerie checkout.
- `tools/slim_rrd.py` — shrinks a raw episode (about 145 MB) to about 6 MB: `python tools/slim_rrd.py data/episode_000052.rrd playground/recordings/stack.rrd`. Raw episodes live in `data/`, which is git-ignored.
- `tools/sim-test.mjs` — runs the self-playing robot headless: `node tools/sim-test.mjs [seed] [seconds]`.

## GitHub Pages

The site lives at the root of the `gh-pages` branch. `.nojekyll` keeps it as plain static files.

```sh
git push origin gh-pages
```

In the repository’s **Settings → Pages**, choose **Deploy from a branch**, branch **gh-pages**, folder **/(root)**. See [GitHub’s publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The custom domain `grachev.ai` is configured separately in Pages settings and DNS.

The Git remote uses the personal SSH alias: `git@github.com-personal:grach0v/personal_webpage.git`.
