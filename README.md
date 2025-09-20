# Overview
This project demonstrates how to perform image clustering using only NumPy, without relying machine learning libraries like scikit-learn. The notebook loads an image, processes it, and applies K-Means clustering algorithm from scratch.

# How It Works?
## Step 1:
the first step is to convert the 3-channel image into a grayscale image with only 1 channel which can be done with opencv's built in function:    
```python
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
## Step 2:
The second step is to extract all pixels at a given y-value and normalize them (convert to floats between 0-1)

![Plot](https://github.com/Lord-Mahdi1383/K-Means-Algorithm-With-Pure-Numpy/blob/main/Plots/Pic%202.png)

```python
horizontal_line = image_gray[y, starting_x:ending_x]
norm_line = horizontal_line / 255.0
```

## Step 3:
it's time to deal with outliers, applying median with a window size of 16 does the job
   
![Plot](https://github.com/Lord-Mahdi1383/K-Means-Algorithm-With-Pure-Numpy/blob/main/Plots/Pic%203.png)

## Step 4: 
the final step is to implement the K-Means algorithm using numpy, these steps repeats until new center points converge or until it the maximum number of iterations (in this case 100) is reached:

![Plot](https://github.com/Lord-Mahdi1383/K-Means-Algorithm-With-Pure-Numpy/blob/main/Plots/Pic%204.png)

   - choose 3 random center points from our data:    
   ```python
   center_points = np.random.choice(data, size=clusters, replace=False)
   ```
   - calculate the distance from each point to each cluster center:    
   ```python
   distance = np.abs(data[:, None] - center_points)    
   labels = np.argmin(distance, axis=1)
   ```
   - calculate new center points for each cluster:    
   ``` python
   new_center_points = np.array([np.mean(data[labels == i]) for i in range(clusters)])
   ```


# Setup
### Clone The Repository
``` bash
git clone https://github.com/Lord-Mahdi1383/K-Means-Algorithm-With-Pure-Numpy.git
cd K-Means-Algorithm-With-Pure-Numpy
```

### Install Requirements
``` bash
pip install -r Requirements.txt
```
