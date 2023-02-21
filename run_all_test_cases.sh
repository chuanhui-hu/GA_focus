#!/bin/bash

for n in {0..9}
do
#   python3 test_GA_focus.py $n Ellipse
#   python3 test_GA_focus.py $n Ellipse _linear
#   python3 test_GA_focus.py $n Ellipse _no_focus
#   python3 test_GA_focus.py $n Ellipse _10gen
 
#   python3 test_GA_focus.py $n Enigma
#   python3 test_GA_focus.py $n Labyrinth
  python3 test_GA_focus.py $n LakeShore
  python3 test_GA_focus.py $n PrimevalIsles
  
done

echo All done

