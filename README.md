# CourseProject

Introduction
------------
As final project for CS 410 Text Information System, we participated in Text Classification Competition to detect Twitter Sarcasm. We were given both Training and Test datasets.

The Training dataset Content
----------------------------
label: SARCASM or NOT_SARCASM
response: The Tweet to be classified
context: The conversation context of the response
example:
{"label": "SARCASM", "response": "@USER @USER @USER I don't get this .. obviously you do care or you would've moved right along .. instead you decided to care and troll her ..", "context": ["A minor child deserves privacy and should be kept out of politics . Pamela Karlan , you should be ashamed of your very angry and obviously biased public pandering , and using a child to do it .", "@USER If your child isn't named Barron ... #BeBest Melania couldn't care less . Fact . 💯"]}

The Test dataset content
------------------------
id: String identifier for sample. This id is required for project submission and grading.
response: The Tweet to be classified
context: The conversation context of the response
example:
{"id": "twitter_1", "response": "@USER @USER @USER My 3 year old , that just finished reading Nietzsche and then asked me : \" ayo papa why these people always trying to cancel someone on Twitter , trying to pretend like that makes them better themselves ? \" . To which I replied \" idk \" , and he just \" cuz hoes mad \" . Im so proud . <URL>", "context": ["Well now that \u2019 s problematic AF <URL>", "@USER @USER My 5 year old ... asked me why they are making fun of Native Americans ..", "@USER @USER @USER I will take shit that didn't happen for $ 100", "@USER @USER @USER No .. he actually in the gifted program and reads on second grade level .  ... and he knows Kansas City is in Missouri"]}

Dataset size statistics
-----------------------
Train	Test
5000	1800

Project Objective
-----------------
Our project objective is to learn from the Training dataset and predict the labels of Test dataset (SARCASM or NOT_SARCASM).


Setup and Usage Instructions
----------------------------
Software Dependencies:
----------------------
1) Python==3.8.3
2) nltk==3.5
3) pandas==1.0.5
4) scikit_learn==0.23.2

Setup and Usage Instructions:
-----------------------------
1) conda create -n "project_demo" python=3.8.3
2) conda activate project_demo
3) git clone https://github.com/subhasishb-coder/CourseProject.git
4) cd CourseProject
5) pip install nltk==3.5
6) pip install pandas==1.0.5
7) pip install scikit_learn==0.23.2
8) cd code
9) python TestClassficationCompetion_Sarcasm_Detection.py

Software Usage Tutorial Link:
-----------------------------
https://mediaspace.illinois.edu/media/t/1_xwb0wmzt


