# Akanksha Gavade — Portfolio Site
    x
A 4-page static site (no build tools needed): `index.html`, `research.html`,
`projects.html`, `experience.html`, sharing `css/style.css` and `js/main.js`.

## How to add / edit text
Open any `.html` file in a text editor. Look for:
- `<!-- EDIT: ... -->` comments — these tell you exactly what to change and where.
- `<div class="editnote">...</div>` boxes — these render on the live page as a
  small dashed box (only you should see these as "notes to self" — **delete
  each one once you've filled it in**, they're not meant to be permanent).

Every card, entry, and timeline item is plain HTML, so you can copy/paste a
whole block (e.g. one `<article class="entry">...</article>`) to add a new
paper or project without touching the CSS.

## Images
Put your own photos in an `/images` folder next to the HTML files, then update
the `src="..."` paths (currently pointing at placeholders like
`images/app_ss.png` or your old GitHub-hosted photo). Recommended: square
photos around 500×500px for the hero image.

## Linking your résumé
Drop a PDF named `AkankshaGavade_Resume.pdf` in the root folder — the "Download
résumé" link on the homepage already points to that filename.

## Deploying to GitHub Pages
1. Copy all these files into your `akankshagavade.github.io` repo (replacing
   the old `README.md`/markdown-based site).
2. Commit and push to the `main` branch.
3. GitHub Pages will serve `index.html` automatically at
   `https://akankshagavade.github.io`.

## Structure at a glance
```
index.html        → About / hero / stats / table of contents / career goals
research.html      → Publications: TOC cards + full abstract entries
projects.html      → Projects: TOC cards + full write-up entries
experience.html    → Timeline: research roles, leadership, education
css/style.css      → All styling (colors, type, layout — edit tokens at top)
js/main.js         → Active-nav highlighting + scroll-reveal animation
```

## Design notes
Palette and type are set as CSS variables at the top of `css/style.css`
(`:root { ... }`) — change `--navy`, `--gold`, `--berry` etc. there to retheme
the whole site at once.
