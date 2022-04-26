import os
import subprocess

from setuptools import find_packages, setup

NAME = "tinyspace"
AUTHOR = f"{NAME} contributors"
URL = "https://github.com/etaoxing/tinyspace"
__version__ = "0.1.0"

install_requires = [
    "numpy",
]

extras_deps = {
    "tests": [
        # Reformat
        "black>=19.10b0",
        # Lint code
        "flake8>=3.7",
        # Find likely bugs
        "flake8-bugbear>=20.1",
        # Run tests and coverage
        "pytest>=5.3",
        "pytest-benchmark>=3.1.0",
        "pytest-order>=1.0.1",
        "pytest-cov",
        "pytest-xdist",
        # # Type check
        # "pytype",
        # Sort imports
        "isort>=5.0",
    ],
    "docs": [
        "sphinx==4.4.0",
        "sphinx-autobuild",
        "myst-parser",
        # # Jupyter notebooks
        # "nbsphinx",
        # For spelling
        "sphinxcontrib-spelling",
        # Type hints support
        "sphinx-autodoc-typehints",
        # Extras
        "sphinx-design",
        "sphinx-copybutton",
        "sphinx-inline-tabs",
        "sphinxcontrib-trio",
        "sphinxext-opengraph",
        # Theme
        "furo",
    ],
}

extras_deps["all"] = [item for group in extras_deps.values() for item in group]


if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()
    cwd = os.path.dirname(os.path.abspath(__file__))
    sha = "Unknown"
    version = __version__

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except subprocess.CalledProcessError:
        pass

    if sha != "Unknown" and not os.getenv("RELEASE_BUILD"):
        version += "+" + sha[:7]
    print("Building wheel {}-{}".format(NAME, version))

    version_path = os.path.join(cwd, NAME, "version.py")
    with open(version_path, "w") as f:
        f.write(f'__version__ = "{version}"\n')
        f.write(f'commit = "{sha}"\n')

    setup(
        name=NAME,
        version=version,
        description="",
        author=AUTHOR,
        url=URL,
        # download_url=f'{URL}/archive/{__version__}.tar.gz',
        license="MIT",
        packages=find_packages(),
        include_package_data=True,
        install_requires=install_requires,
        extras_require=extras_deps,
        python_requires=">=3.7",
        zip_safe=False,
    )
