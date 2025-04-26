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
### 6. Controls for the annotator (Stage 1 - cell labeling)

In the first stage, draw rectangles around all the cells of interest. 
You can scroll though the z depths in two ways:
1. The scroll bar at the top of the window
2. The ```w``` key for up and the ```s``` key for down

When you are drawing rectangles press ```enter``` to save. 
If you want to clear the rectangles to start over, press ```c```.

### 7. Controls for verifying the cell slices (Stage 2)

In the second stage, you go through the z depth of every cell you drew a rectangle around in the first stage. 
You need to verify that the cell has the red nucleus and green dot in the image, and is cropped well.

Press ```g``` for good (yes it has the nucleus and green dot), and ```b``` for bad (it does not fit the requirements). 

Pressing either key will move on to the next image, and everything is automatically saved.