# SAFER on your F1TENTH car

Two controllers for real_track: **PPO** (the expert) and **SAFER** (our planner). They drive the centre line at
1.0 m/s. See `videos/` for SAFER in simulation.

**Setup:** give this folder to an AI coding assistant (e.g. Claude Code) on the computer that will drive the car, and
ask it to follow `AI_SETUP.md`. SAFER needs a CUDA GPU.

**Test 1 (now)**
1. Put the car on the track and start the localisation. Run `python3 -m safer_deploy.check_setup`. If any line is not
   PASS, send the output to Yansong before driving.
2. PPO, 2-3 runs of about 2 minutes each: `bash scripts/run.sh ppo ppo_1`. Ctrl-C or your dead-man switch stops it.
3. SAFER, the same way, until there are 15-30 minutes of SAFER driving in total: `bash scripts/run.sh safer safer_1`.
4. Keep the dead-man switch in hand. Stop if the car leaves the lane, touches a wall or weaves. No intentional crashes.
   Taking over with the joystick is fine: it is recorded automatically. Also note it in the run sheet.
5. Write one line per run in `RUN_SHEET.csv` and film a few laps.
6. Run `bash scripts/pack_runs.sh test1`, then send Yansong the archive link and the videos.

**Test 2 (later):** we send a fine-tuned SAFER checkpoint. Your AI assistant switches to it (see `AI_SETUP.md`). Repeat
steps 1-6 and film the demo.

**Notes (please read first).** The policies were trained for your simulator's setting. If your real car differs in
any of these points, tell Yansong before driving.
- The real track is `data/real_track.png` at 0.05 m per pixel, and your localisation runs on this map (same origin).
- The LiDAR has 1080 beams over 270 deg and is mounted 0.275 m ahead of `base_link`.
- The car takes `AckermannDriveStamped` commands (speed in m/s, steering in rad, at most 0.4189 rad) and drives 1.0 m/s.
- The car runs ROS 2 Humble.
- Where SAFER runs: it needs a CUDA GPU. If the car's computer has one, the node runs on the car. Otherwise it runs on
  a GPU workstation on the car's ROS 2 network: the workstation receives the sensor topics, computes the action and
  sends `/drive` back. Check that this fits your setup; if not, your AI assistant can adapt the code (see `AI_SETUP.md`).
