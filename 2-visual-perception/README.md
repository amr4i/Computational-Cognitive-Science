# Visual Perception
The simulation of how neurons work to provide visual perception.

_Authors_:
1. Amrit Singhal
2. Mrinaal Dogra
3. Aarsh Prakash Agarwal

### Part 1: Designing a complex cell

_Purpose_: 
1. To create an image containing the provided shape, named `test.png`.
2. To create all the required Gabor filters (simulating the simple cells) and detect the shape using output of all these Gabor filters together (simulating a complex cell) succesfully.


_Usage_: `python q1.py <shape>`
	where `shape` can be `s` for sqaure, or `t` for traingle, which is the shape that is created. 

_Output_: A single image `output.png` which is the output of the gabor filters for the image.


### Part 2: Creating images for Feature Search or Conjunction Search

_Pupose_:
Create an image having multiple objects of different shape (traingle or square) and different color(red or blue) depending the search paradigm provided. 
If _feature search_, then all objects have the same value for one feature(either color or shape) and one of them differs in the other feature, thus resulting in the pop-out effect.
If _conjunction search_, then the odd-one-out object matches some other objects in shape, and the remaining other objects in color, but no other object has the same shape+color combination.

_Usage_: `python q2.py <num_objects> <paradigm>`
	where, `<num_objects>` is the number of objects to be placed in the image, and `<paradigm>` is the search paradigm for which the image has to be designed. It can be `f` for _feature search_, or `c` for _conjunction search_.

_Output_: A single image named according to the `<paradigm>` and `<num_objects>` chosen. 


### Part 3: Simulating a feature integration theory

_Purpose_:
To create a feature integration theory simulation, simulating feature search and conjunction search, and show the comparision between them.

_Usage_: `python q3.py`

_Output_:
1. A folder `images` containing all the different input images created for different search and num_objects combinations.
2. A folder `feature_maps` that holds the four feature maps for _red_, _blue_, _square_ and _triangle_, for each of the input images, marking locations where each feature is present.
3. A folder `results` showing the output of the search paradigms for each fo the input images.
4. A output plot `plot.png` comparing the response time for feature search and conjunction search.

(_Note_: Due to the randomness involved in the various steps, the plot generated may not always look same. Running the code a few times will most likely give a similar simulation though.)