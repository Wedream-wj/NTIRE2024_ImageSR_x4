# Step 1, run RepRLFN
python test_demo_reprlfn.py
# Step 2, download weight: https://drive.google.com/file/d/1vMTWLvrn9KpGDVTGyn0RB2blBbNsQYJe/view?usp=sharing, and place it in ./model_zoo/
python test_demo_hat.py
# Step 3, model ensemble
python ensemble_results.py

