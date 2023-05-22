### Useful installation steps if running the notebook locally, using a virtual environment

## Dataset
Due to the authors' requirement, the dataset cannot not be made publicly available. However, if you wish to gain access to the dataset, you can contact me at a.stoica@student.rug.nl .


## Build and run the project

1. Navigate to the directory where you want to store the project.
```bash
cd your/directory/for/the/project
```

2. Clone the repository.
```bash
git clone https://github.com/andreistoica12/research-internship.git
```

3. Navigate to the root directory of the project.
```bash
cd research-internship/
```

4. Switch to the covaxxy branch.
```
git checkout covaxxy
```

5. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install `virtualenv`.
```bash
pip install virtualenv
```

6. Create a virtal environment named `venv`.
```bash
python3 -m venv venv
```
7. Activate the virtual environment.
```bash
source venv/bin/activate
```

8. Install all the dependencies from the `requirements.txt` file into your virtual environment.
```bash
pip install -r requirements.txt
```

9. Select the virtual environment you just created as the kernel (from VS Code). Easily run and debug cells.



<!-- TODO: adapt commands for Windows and check if htey are correct for both Linux and Windows -->