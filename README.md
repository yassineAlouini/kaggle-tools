<h1>Kaggle tools</h1>

<h2>About</h2>

This is a repository containing tools, scripts and code snippets that I often
use in Kaggle challenges and more generally in data science and machine learning
tasks.

<h2>Installation (Linux and OS X)</h2>

To install the project dependencies, run:

```pip install -r requirements```

If you are familiar with [**Conda**](!http://conda.pydata.org/docs/),
I would suggest creating a virtual environnement
and installing the dependencies in the following fashion:

```
conda create --name kaggle-tools --file requirements.txt

```

Then activate the environment with the following command:

```
source activate kaggle-tools
```

If you want to process images, you need to install [**OpenCV**](!http://opencv.org/). Here is the easiest
way I found to do it (activate the virtual environment first):

```
conda install -c https://conda.binstar.org/menpo opencv3
```

<h2>Roadmap</h2>

I am planning to release an alpha version (v0.0.3) by the end of 2018. Check the ROADMAP.md file for more details.

<h2>Contributing</h2>

Feel free to send a link to tools that you use or make a pull request.


<h2>License</h2>

The MIT License (MIT)

Copyright (c) 2018 Yassine Alouini
