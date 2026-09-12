import os
import shutil
import subprocess

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildPy(build_py):
    def run(self):
        subprocess.check_call(["make", "libtinyfin.so"])
        super().run()
        package_dir = os.path.join(self.build_lib, "tinyfin")
        os.makedirs(package_dir, exist_ok=True)
        shutil.copy2("libtinyfin.so", os.path.join(package_dir, "libtinyfin.so"))


setup(cmdclass={"build_py": BuildPy}, distclass=BinaryDistribution)
