# Render music

The render window can loop a short tune while an episode plays. It is disabled
by default; enable it with `RENDER_MUSIC` in `config/gym_env/render_params.yaml`.

By default a few short original riffs are synthesised (`cruise`, `drive`,
`open_road`). To play your own audio instead, add a file named after the track
here, for example:

    assets/music/cruise.mp3

`.mp3`, `.ogg`, and `.wav` are supported. Audio files are not tracked by git —
only add material you have the right to use.
