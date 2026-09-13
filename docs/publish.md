# Publishing our lab

The migration target is the new `MoE-Scope` repository. Our code, tests, journal,
and raw A100 evidence travel together. Historical evidence keeps its original
timestamps, source archive, hashes, and original Git revision.

The Astro site builds for `/MoE-Scope/`. Navigation, Markdown links, JavaScript
trace downloads, generated assets, and canonical URLs share this base. For a
custom domain, set `SITE_BASE=/` and `SITE_ORIGIN` to the HTTPS origin when building.

## One-time GitHub setup

1. Give our GitHub connection write access to the new repository.
2. Import the committed project into its `main` branch, preserving existing work.
3. Under **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
4. Run **Publish the lab** from Actions, or push a change to `main`.

The workflow installs pinned project dependencies, checks Python behavior and
the evidence, exports the website data, checks the browser compiler, builds the
site, and verifies links before uploading a Pages artifact. A separate deployment
job receives only Pages and OIDC write permissions. No Cloudflare token or GPU
credentials are required.

The Pages action provides the actual owner origin and repository base path at
build time. The deployment job reports the published URL. A successful local
build alone does not mean the site has been published.

## Local verification

```bash
python -m pip install -e . pytest==8.4.2
python -m pytest
python scripts/export_web.py
npm --prefix web ci
npm --prefix web test
npm --prefix web run check
npm --prefix web run build
python scripts/check_site.py
npm --prefix web run dev
```

Open `http://localhost:4321/MoE-Scope/`. Check the evidence filters, a captured
trace, its JSON download, and the links in a journal entry.

Official deployment references: [Astro on GitHub Pages](https://docs.astro.build/en/guides/deploy/github/)
and [GitHub Pages publishing configuration](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
