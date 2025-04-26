# CVSE_cell_labeling

## Set up instructions

### 1. Clone the repository
In your terminal, in the location you want this folder, run:

```
git clone https://github.com/anyazorin/CVSE_cell_labeling.git
```

### 2. Put your data into the ```data``` folder
These should be .dv files. It is assumed they have z stack and no time stack.

### 3. In terminal, navigate to the repository
```
cd CVSE_cell_labeling
```

### 4. Install requirements

```
pip install -r requirements.txt
```

### 5. Run the data labeling
```
python run_annotator.py -i [your image name here]
```