from setuptools import setup, find_packages

setup(
    name="u20cam_720p",
    version="0.0.0",
    packages=find_packages(),
    py_modules=["calibrate", "undistort", "screenshot", "test", "trial"],
)
