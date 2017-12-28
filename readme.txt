
This is the model and testing code for the iccv 2017 paper: Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition

To test the model, you should finish the following two steps:


1)download the data

You can download the three datasets from the official site:

	for the bird dataset: 		http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

	for the car dataset:  		http://ai.stanford.edu/~jkrause/cars/car_dataset.html

	for the aircraft dataset:	http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

And then copy the images to the "bird_data" ("car_data", "air_data") folder. the path should fit the ./bird_data/test_list.txt (./car_data/test_list.txt, ./air_data/test_list.txt) well. 


2)install pycaffe

As for the MA-CNN model uses "Transpose" Layer Type, the standard caffe is not runable. The transpose layer can be added by referring https://github.com/houkai/caffe  

For windows users, we also provided the compiled pycaffe files in the "caffe" folder, you can just copy this folder to you-python-path/Lib/site-packages. 

After installed pycaffe, you can simplly run 
	
	python bird.py

to test the model.



model accuracy

	bird accuracy:	86.58
	car accuracy:	92.75
	air accuracy:	90.00


note: the bird prototxt is noted which can help to undersdand the structrue of MA-CNN
