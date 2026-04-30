# Exercise 1 Solution – Box Detection

This folder contains a complete solution for **Exercise 1.1** and material for **Exercise 1.2** from the FAU Computer Vision Project. The implementation follows the exercise sheet: load the `.mat` files, visualize the data, implement **RANSAC for planes**, detect the **floor**, detect the **top plane of the box**, keep the **largest connected component**, and estimate the **box dimensions** from the final result. The exercise requires a self-written RANSAC and explicitly says not to use a ready-made RANSAC implementation. The provided script does exactly that. 

## Files included

- `box_detection_solution.py`  
  Complete Python solution.

- `discussion.md`  
  Ready-to-submit discussion for Exercise 1.2.

- `results/summary.json`  
  Measured results for the 4 provided examples.

- `results/example*_inputs.png`  
  Visualization of amplitude image, distance image, and subsampled point cloud.

- `results/example*_masks.png`  
  Visualization of floor mask, filtered floor mask, raw top-plane mask, and final box top.

- `results/example*_3d.png`  
  3D visualization of the scene, floor, box top, and estimated corners.

## Required Python packages

Install these if needed:

```bash
pip install numpy scipy matplotlib
```

The exercise sheet mentions `numpy`, `scipy`, `scikit-learn`, and `matplotlib` as useful libraries. This solution uses `numpy`, `scipy`, and `matplotlib`. It does **not** use scikit-learn for RANSAC, which matches the assignment requirement. 

## How to run

Put the script in the same folder as these files:

- `example1kinect.mat`
- `example2kinect.mat`
- `example3kinect.mat`
- `example4kinect.mat`

Then run:

```bash
python box_detection_solution.py
```

It will create a `results/` folder and save all output figures plus a JSON summary.

## Method overview

The method is directly based on the exercise description:

1. Load one of the example files and visualize amplitude image, distance image, and 3D point cloud. 
2. Ignore invalid points whose z-value is `0`. 
3. Run self-written RANSAC on the 3D point cloud to detect the **floor plane**. 
4. Convert floor inliers into a mask image.
5. Improve the mask using morphological filtering. The exercise explicitly asks to evaluate which operators work best. In this solution, **closing** is used first to fill small holes in the floor mask, then **opening** is used to remove small noise blobs. 
6. Remove the floor points and run RANSAC again on the remaining 3D points to get the **top plane of the box**. 
7. Use connected components and keep the **largest connected component**, because the sheet says it can be assumed to delimit the box top. 
8. Estimate the **height** using the distance between top-plane points and the floor plane, with plane offset as a cross-check.
9. Estimate **length** and **width** from a PCA-aligned oriented bounding box on the top plane.
10. Visualize masks, planes, and corners. The exercise asks for a simple visualization of the results. 

## Important implementation choices

### 1. Plane representation
The script uses a normalized plane representation:

- Plane normal: `n`
- Plane equation: `n · x = d`

This is the same representation recommended by the exercise.

### 2. Why batch RANSAC?
The point cloud is fairly large, so evaluating one hypothesis after another can be slow. The assignment also says to use NumPy efficiently and avoid unnecessary loops. This solution therefore evaluates many RANSAC hypotheses in one batch using matrix operations. 

### 3. Why connected components?
The raw top-plane mask may still include points that lie on other planar regions. Keeping only the largest connected component usually removes these spurious regions and gives a clean box top. This follows the exercise statement directly.

## Measured results for the provided examples

The script produced the following values on the uploaded example files:

| Example | Length (m) | Width (m) | Height (m) |
|---|---:|---:|---:|
| 1 | 0.649 | 0.470 | 0.189 |
| 2 | 0.472 | 0.402 | 0.189 |
| 3 | 0.474 | 0.399 | 0.192 |
| 4 | 0.470 | 0.374 | 0.184 |

The script also stores a second height estimate in `summary.json` using the plane offset difference as a consistency check.

## Notes for presentation

When you show this to the advisor, explain the flow like this:

- First RANSAC → floor
- Morphology → better floor mask
- Remove floor
- Second RANSAC → box top
- Largest connected component → final box top
- Plane distance + corners → dimensions


