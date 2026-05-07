# Exercise 1.2 – Discussion

## Result visualization

The result is visualized with:

- the amplitude image
- the distance image
- a subsampled 3D point cloud
- the raw floor mask
- the filtered floor mask
- the raw box-top mask
- the largest connected component of the box top
- a 3D view with the estimated box corners

This is enough to understand whether the algorithm works correctly and also makes debugging easier.

## Why the method works

The main idea is that the floor and the top of the box are both approximately planar. Instead of using single noisy depth values, the algorithm fits complete planes to many 3D points. This makes the height estimation much more stable. The floor is usually the dominant plane in the whole scene, and after removing it, the top of the box is the dominant remaining plane that is most relevant for the object.

Morphological filtering helps because the raw RANSAC inlier mask is not always clean. Small holes, isolated noise pixels, or little broken regions can appear. Closing removes holes and opening removes tiny isolated regions. After that, connected components make it possible to keep only the region that really belongs to the box top.

## Weaknesses of this simple implementation

### 1. Strong dependence on thresholds
The algorithm depends on the RANSAC inlier threshold and the morphology kernel sizes. If the threshold is too small, the plane breaks apart. If it is too large, points from other objects may be included.

### 2. Assumption that the floor is the largest plane
This is reasonable in the provided examples, but it may fail in a more complex scene. For example, a large wall or background plane could dominate the point cloud.

### 3. Assumption that the second important plane is the box top
This is explicitly true for the exercise examples, but not always true in general scenes. Another planar object could be detected instead.

### 4. Limited robustness to missing depth values
Invalid measurements are removed by ignoring points with z = 0. That is correct and necessary, but large missing regions can still weaken the detection.

### 5. Bounding-box estimation on the top plane is still approximate
The final length and width depend on the visible top region. If the top is partially occluded or the segmentation is slightly too large or too small, the estimated dimensions change.

### 6. Runtime still grows with point-cloud size
The implementation is vectorized and reasonably fast, but RANSAC can still be expensive for larger point clouds or many scenes.

## Possible improvements

### 1. Adaptive thresholds
Instead of using fixed thresholds, one could estimate the noise level from the data and set the inlier threshold automatically.

### 2. Better plane ranking
After removing the floor, multiple candidate planes could be estimated and ranked not only by size but also by:
- parallelism to the floor
- height above the floor
- compactness of the component
- rectangularity of the top mask

This would make the method safer in cluttered scenes.

### 3. Use of surface normals
Local surface normals could be estimated before RANSAC. Then one could restrict the search to planes with normals compatible with the floor direction.

### 4. Region-of-interest restriction
The box is usually located above the floor and inside a meaningful image region. Restricting the search space could reduce false detections and improve speed.

### 5. Better corner extraction
Instead of using a PCA-aligned bounding box, one could fit 2D lines to the boundary of the top component and intersect them. That would provide more geometrically precise corners.

### 6. Faster RANSAC variants
The lecture and slide deck also mention extensions such as MLESAC and preemptive RANSAC for the individual task. These could improve robustness or speed if implemented carefully. The lecture slides explicitly list them as further tasks for the individual exercise. fileciteturn0file1L51-L55

## Conclusion

For the provided data, the pipeline works well:
- RANSAC finds the floor plane robustly,
- morphology improves the mask,
- a second RANSAC finds the box top,
- connected components isolate the actual box region,
- and the final plane and corners are sufficient to estimate the box dimensions.

So even though the method is simple, it is already a good baseline and matches the workflow requested in the exercise sheet.
