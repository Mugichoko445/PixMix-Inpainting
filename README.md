A C++ implementation of PixMix inpainting.

[Jan Herling and Wolfgang Broll, "High-Quality Real-Time Video Inpaintingwith PixMix," Vol. 20, Issue 6, pp. 866 - 879, 2014.](http://ieeexplore.ieee.org/document/6714519/)

# Features
* Inpainting only
	* No object selection and tracking
	* Only the basic spatial and appearance cost minimization (i.e., eq. 2 to 6)
* Portable and compatible implementation
	* The code requires a C++ complier and [OpenCV (2.X or heigher)](https://opencv.org/) only
* A collaborative work of [Shohei Mori](http://hvrl.ics.keio.ac.jp/mori/) and [Mariko Isogawa](https://sites.google.com/site/marikoisogawa/home/eng).
* Lisence
	* Lisence free but limited to **research purpose only**
	* Note that the code is **NOT** the original implementation (i.e., results may be different from the ones in the original paper)
	* We, therefore, assume no responsibility if a problem occurs
	* BUT we are pleased to have any kinds of feedback from you!!

# Usage
* Inputs
	* MAGENTA_MASK_MODE
		* 3ch color image with magenta color mask
	* The other
		* 3ch color image
		* 1ch black and white mask image
* Output
	* 3ch inpainted image

# Example Results

|color|mask|result|
|:-|:-|:-|
|![birds](./data/birds.png)|![birds_mask](./data/birds_mask.png)|![birds_res](./data/birds_res.png)
|![colosseum](./data/colosseum.png)|![colosseum_mask](./data/colosseum_mask.png)|![colosseum_res](./data/colosseum_res.png)
|![firenze](./data/firenze.png)|![firenze_mask](./data/firenze_mask.png)|![firenze_res](./data/firenze_res.png)
|![frog](./data/frog.png)|![frog_mask](./data/frog_mask.png)|![frog_res](./data/frog_res.png)
|![graffiti](./data/graffiti.png)|![graffiti_mask](./data/graffiti_mask.png)|![graffiti_res](./data/graffiti_res.png)
|![graz](./data/graz.png)|![graz_mask](./data/graz_mask.png)|![graz_res](./data/graz_res.png)
|![graz_wall](./data/graz_wall.png)|![graz_wall_mask](./data/graz_wall_mask.png)|![graz_wall_res](./data/graz_wall_res.png)
|![schlossberg_statue](./data/schlossberg_statue.png)|![schlossberg_statue_mask](./data/schlossberg_statue_mask.png)|![schlossberg_statue_res](./data/schlossberg_statue_res.png)
|![schlossberg_tower](./data/schlossberg_tower.png)|![schlossberg_tower_mask](./data/schlossberg_tower_mask.png)|![schlossberg_tower_res](./data/schlossberg_tower_res.png)
|![venice_roof](./data/venice_roof.png)|![venice_roof_mask](./data/venice_roof_mask.png)|![venice_roof_res](./data/venice_roof_res.png)
|![venice_river](./data/venice_river.png)|![venice_river_mask](./data/venice_river_mask.png)|![venice_river_res](./data/venice_river_res.png)
|![venice_wall](./data/venice_wall.png)|![venice_wall_mask](./data/venice_wall_mask.png)|![venice_wall_res](./data/venice_wall_res.png)


# Tested Environment List
* Win 10 64bit + VS2015 + OpenCV 3.3.1
* maxOS 10.13.3 + OpenCV3.3.1