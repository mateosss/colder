# COLDER: Colmap exporter from Blender

## Overview videos

- first versions usage: <https://youtu.be/kqH3B4dpb1E>
- vrstreet example: <https://youtu.be/SkQsm3qZl3U>
- kingshall example: <https://youtu.be/jZ_KDPutZWY>

## CLI Usage

```bash
uv run python cli.py setup_blender # Install dependencies into Blender's Python environment
uv run python cli.py generate main.blend configs/default.json # Generate dataset from main.blend using default.json
```

## Blender UI Usage

TODO

## Creation of realistic dataset

This should be automatized in the future, for now:

- Get good model: e.g., sketchfab or blenderkit addons. (e.g., https://sketchfab.com/aurelien_martel)
- Create single camera with spawn camera (so that all the intrinsics are created)
- Add HDRI if not yet: go to Shading workspace -> world shader -> add environment image -> go to blender folder for hdris like in the viewport
- Walk around with the camera while recording (press red button, and spacebar to start animation)
- To add handheld realistic animation noise: go to graph editor, select XYZRPY components of the animation and add noise to each (lookup in google for good values)
- Then press the Spawn Animation Cameras button
- Run generate_all button or `cli.py generate` as usual

