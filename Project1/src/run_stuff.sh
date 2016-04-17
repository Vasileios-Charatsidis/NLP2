#!/bin/bash
# All models that need to be run are available here
# Make sure the paramaters are in order french english french english
# run as "sh run_stuff.sh" or select just pick one and run it in a terminal

echo "Training IBM1"
python main.py ibm1 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm1.pkl > ibm1.log

#echo "Training IBM1 with null words"
#python main.py ibm1_add0 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm1_add0.pkl > ibm1_add0.log

#echo "Training IBM1 with smoothing"
#python main.py ibm1_smooth ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm1_smooth.pkl > ibm1_smooth.log

#echo "Training IBM2 uniform"
#python main.py ibm2 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm2_uniform.pkl uniform > ibm2_uniform.log

#echo "Training IBM2 random1"
#python main.py ibm2 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm2_random1.pkl random > ibm2_random1.log

#echo "Training IBM2 random2"
#python main.py ibm2 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm2_random2.pkl random > ibm2_random2.log

#echo "Training IBM2 random3"
#python main.py ibm2 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm2_random3.pkl random > ibm2_random3.log

echo "Training IBM2 ibm1"
python main.py ibm2 ../NLP2_Project1_data/training/hansards.36.2.f ../NLP2_Project1_data/training/hansards.36.2.e ../NLP2_Project1_data/testing/test/test.f ../NLP2_Project1_data/testing/test/test.e ../NLP2_Project1_data/testing/answers/test.wa.nonullalign 15 ibm2_ibm1.pkl ibm1 ibm1.pkl > ibm2_ibm1.log

