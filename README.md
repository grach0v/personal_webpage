# grachev.ai

Denis Grachev’s personal homepage. Plain HTML and CSS. No build step or frontend dependencies.

## Preview

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. Edit `index.html` and `style.css`.

## GitHub Pages

The site lives at the root of the `gh-pages` branch. `.nojekyll` keeps it as plain static files.

```sh
git push origin gh-pages
```

In the repository’s **Settings → Pages**, choose **Deploy from a branch**, branch **gh-pages**, folder **/(root)**. See [GitHub’s publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The custom domain `grachev.ai` is configured separately in Pages settings and DNS.

The Git remote uses the personal SSH alias: `git@github.com-personal:grach0v/personal_webpage.git`.
