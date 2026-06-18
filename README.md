### Setup a conda env

```bash
conda create --name u20cam python=3.11
conda activate u20cam
```

Install the repo as editable mode, and install extra packages listed in `requirements.txt`:

```bash
cd ~/u20cam_720p
pip install -e . -r requirements.txt
```

### Single camera

Connect the U20Cam 720P in a USB socket. Run a demo to test the camera and undistortion function:
```bash
conda activate u20cam
cd ~/u20cam_720p
python trial.py
```

### Multiple cameras

Launch wrist camera:

```bash
conda activate u20cam
cd ~/u20cam_720p
python multi_cameras.py --wrist
```

Launch side camera:

```bash
conda activate u20cam
cd ~/u20cam_720p
python multi_cameras.py --side
```

Launch both wrist and side cameras:

```bash
conda activate u20cam
cd ~/u20cam_720p
python multi_cameras.py --side --wrist
```