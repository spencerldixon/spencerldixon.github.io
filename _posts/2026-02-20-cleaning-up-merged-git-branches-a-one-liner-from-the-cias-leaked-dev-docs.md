---
layout: post
title: "Cleaning up merged git branches: a one-liner from the CIA's leaked dev docs"
description: "How to delete all merged git branches locally with a single command. This one-liner has been in my zshrc since 2017 — I found it buried in the CIA's Vault7 leaked developer docs."
date: 2026-02-20 00:00 +0000
blog: true
tags: [git, productivity, tools]
---

In 2017, WikiLeaks published Vault7 - a large cache of CIA hacking tools and internal documents. Buried among the exploits and surveillance tools was something far more mundane: [a page of internal developer documentation with git tips and tricks](https://wikileaks.org/ciav7p1/cms/page_1179773.html).

Most of it is fairly standard stuff, amending commits, stashing changes, using bisect. But one tip has lived in my `~/.zshrc` ever since.

## The Problem

Over time, a local git repo accumulates stale branches. Every feature branch, hotfix, and experiment you've ever merged sits there doing nothing. `git branch` starts to look like a graveyard.

You can list merged branches with:

```bash
git branch --merged
```

But deleting them one by one is tedious. The CIA's dev team has a cleaner solution:

## The original command

```bash
git branch --merged | grep -v "\*\|master" | xargs -n 1 git branch -d
```

How it works:

- `git branch --merged` — lists all local branches that have already been merged into the current branch
- `grep -v "\*\|master"` — filters out the current branch (`*`) and `master` so you don't delete either
- `xargs -n 1 git branch -d` — deletes each remaining branch one at a time, safely (lowercase `-d` won't touch unmerged branches)


## The updated command

Since most projects now use `main` instead of `master`, you can update the command and exclude any other branches you frequently use:

```bash
git branch --merged origin/main | grep -vE "^\s*(\*|main|develop)" | xargs -n 1 git branch -d
```

Run this from `main` after a deployment and your branch list goes from 40 entries back down to a handful.

I keep this as a git alias so I don't have to remember the syntax:

```bash
alias ciaclean='git branch --merged origin/main | grep -vE "^\s*(\*|main|develop)" | xargs -n 1 git branch -d'
```

Then in your repo just run:

```bash
ciaclean
```

Small thing, but one of those commands that quietly saves a few minutes every week and keeps me organised.
