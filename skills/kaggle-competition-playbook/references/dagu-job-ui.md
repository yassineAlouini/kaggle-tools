# One web UI for long-running and scheduled jobs (Dagu)

**Problem (2026-11, one workstation shared by many competitions):** jobs were
spread across tmux sessions, `systemd-run` units, systemd timers and per-repo queue
scripts. Answering "what is running, what is planned" meant checking each one.

**Fix:** every long or scheduled job goes through [Dagu](https://github.com/dagucloud/dagu).
Dagu is a single Go binary with a web UI, cron schedules, named queues, history and
per-step logs. It needs no database or message broker.

**Verified on the host (2026-11-07, Dagu v2.25.8):**
- A systemd timer job (the RSNA Knee writeup watcher) was migrated and run with a
  `--dry-run` parameter. The watcher state was unchanged apart from its timestamp.
- A smoke test went through the `gpu` queue.
- The API returns 401 without a login.

## Setup (once per host)

1. Install a release that has been out for a few days and check the release's
   `checksums.txt`:
   `gh release download vX.Y.Z -R dagucloud/dagu -p 'dagu_*_linux_amd64.tar.gz' -p checksums.txt`,
   then `sha256sum -c`, then `install -m755 dagu ~/.local/bin/`.
2. Any first `dagu` command creates `~/.config/dagu/{base.yaml,dags/example-*.yaml}`.
   Delete the examples. `base.yaml` defaults to `overlap_policy: skip` and
   `catchup_window: "6h"`: after downtime, missed cron runs inside that window are
   replayed.
3. Write `~/.config/dagu/config.yaml`. Dagu runs arbitrary shell commands, so bind it to
   localhost and keep a login on:
   ```yaml
   host: 127.0.0.1
   port: 8080
   tz: Europe/Paris
   default_shell: /bin/bash
   check_updates: false
   terminal: {enabled: false}
   auth:
     mode: builtin
     builtin:
       initial_admin: {username: admin, password_file: ~/.config/dagu/admin_password}  # chmod 600
       token: {secret_file: ~/.config/dagu/jwt_secret, ttl: 720h}
   queues:
     enabled: true
     config:
       - {name: gpu, max_concurrency: 1}     # one GPU job at a time on a single card
       - {name: cpu, max_concurrency: 4}
       - {name: light, max_concurrency: 8}   # Kaggle API polls, small scripts
   ```
   Write absolute paths in the file; the `~` above is shorthand.
4. Create the systemd user service `~/.config/systemd/user/dagu.service`. Set an
   explicit `PATH`, because the user manager's environment has no conda.
   `KillMode=process` keeps running jobs alive across a Dagu restart.
   ```ini
   [Service]
   ExecStart=%h/.local/bin/dagu start-all --config %h/.config/dagu/config.yaml
   Environment=PATH=%h/.local/bin:%h/miniconda3/envs/kaggle/bin:/usr/local/bin:/usr/bin:/bin
   Restart=on-failure
   KillMode=process
   [Install]
   WantedBy=default.target
   ```
   Run `systemctl --user enable --now dagu`. Check that lingering is on
   (`loginctl show-user $USER -p Linger`) so the service runs without a login session.

## Agent recipe (every job longer than ~10 min, and every scheduled job)

```yaml
# ~/.config/dagu/dags/<competition>-<job>.yaml
description: <what it does and why: one line>
group: <competition>                 # one group per competition in the UI
tags: [<competition>, train]         # train | infer | eval | kaggle-api | scheduled
queue: gpu                           # gpu | cpu | light
working_dir: /abs/path/to/<competition>
timeout_sec: 36000
# schedule: "0 9 * * *"              # cron in config tz; recurring jobs only
params:
  - DRY_RUN: ""                      # optional; reference it as ${DRY_RUN}
steps:
  - name: train
    run: /abs/path/to/python train.py --cfg cfg.yaml ${DRY_RUN}
```

- `dagu validate <file>`, then `dagu enqueue <name>` (add `-p "DRY_RUN=--dry-run"` for
  a test). `enqueue` hands the run to the service, so it outlives the agent session.
  `dagu start` and `dagu exec` run in the calling shell.
- Monitor with `dagu status <name>`, `dagu history <name>` and `dagu stop <name>`.
  `dagu status` prints the `run:` line with `${PARAM}` unexpanded. Check the expanded
  arguments in the run's `status.jsonl` under `~/.local/share/dagu/data/dag-runs/`.
- Jobs run inside `dagu.service`'s cgroup, which has no memory limit by default. For a
  per-job cap, wrap the step as
  `systemd-run --user --scope -p MemoryMax=<measured>G ...`.
- `dagu rm -H -d <name>` asks for confirmation. Pipe `y` to it in non-interactive
  shells.

## Migrating a systemd timer

1. Copy `WorkingDirectory`, `ExecStart`, the environment and the timeout into a job file.
   The `OnCalendar` value becomes `schedule:`.
2. Enqueue it once in a harmless mode. If the job has side effects (submits, PRs,
   messages to other agents), use a dry-run flag. Compare its outputs with the timer's
   last run.
3. Run `systemctl --user disable --now <timer>`. Keep the unit files so the timer can be
   re-enabled to roll back.

## What it does not cover

- **Agent sessions:** use `claude agents` (Claude Code Agent View).
- **Research plans:** files such as ml-ralph's `kanban.json` are not jobs.
- **Existing queues:** a working per-repo queue (for example a lane-based `jobq.py` in
  the middle of a campaign) can stay as it is. Move it at a natural break, not mid-run.
