# Kaggle tools

## About

This is a repository containing tools, scripts and code snippets that I often
use in Kaggle challenges (more generally in data science and machine learning
tasks).

## Installation (Linux and OS X)

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

## Contributing

Feel free to send a link to tools that you use or make a pull request.


## License

The MIT License (MIT)

Copyright (c) 2016 Yassine Alouini
