# Publishing our lab on Vercel

The `KingReaper6940/MoE-Scope` repository contains our code, ten-entry journal,
and raw NVIDIA A100 evidence. Historical evidence keeps its original timestamps,
source archive, hashes, and model revision.

The Astro site builds at the domain root. Navigation, Markdown links, JavaScript
trace downloads, generated assets, and canonical URLs use that same base. Vercel
supplies `VERCEL_PROJECT_PRODUCTION_URL`, which Astro uses for canonical URLs.

## Import the repository

1. In Vercel, choose **Add New → Project** and import
   `KingReaper6940/MoE-Scope`.
2. Leave the root directory at the repository root.
3. Deploy. The committed `vercel.json` selects Astro and defines the install,
   build, and output settings.

Vercel runs `npm --prefix web ci --no-audit --no-fund`, builds with
`npm --prefix web run build`, and serves `web/dist`. The generated website data
is committed, so deployment does not require Python, Runpod, model weights, or
GPU credentials.

For a custom domain, add it in the Vercel project after the first deployment.
No environment variables are required. `SITE_ORIGIN` remains available as an
explicit canonical-origin override.

## Refreshing evidence later

When the versioned evidence changes, regenerate and verify the committed web data:

```bash
python scripts/export_web.py
python -m pytest
npm --prefix web test
npm --prefix web run check
npm --prefix web run build
python scripts/check_site.py
```

The repository CI performs this full source-to-site validation. A Vercel build is
deliberately smaller because it deploys the already reviewed data snapshot.

Official references: [Astro on Vercel](https://vercel.com/docs/frameworks/frontend/astro)
and [Vercel build configuration](https://vercel.com/docs/builds/configure-a-build).
