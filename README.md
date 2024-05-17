# Entity-Linking-of-Clinical-Notes

This was performed as a capstone project for completion of a Data Science Master's Degree at Hofstra University.  The project was based off of the competition at https://www.drivendata.org/competitions/258/competition-snomed-ct/page/816/ .

This entity linking procedure follows a 2-step process:
1. NER
2. Linking

For NER, two labeling schemes were tested.  To produce a trained NER model for either labeling scheme and output results for the test set, run the 3Label_NER_Final.ipynb or 7Label_NER_Final.ipynb file.

For Linking, the 3 label scheme produces one linking model and the 7 label scheme produces 3 separate linking models.  Two different loss functions were tested as well.  To produce trained Linking models and output results for the test set:
* Run Linking_3Label.ipynb: 3 label linker model, Constrastive Loss (only synonym pairings left uncommented)
* Run Linking_7Label.ipynb: 7 label linker model, Constrastive Loss (only synonym pairings left uncommented)
* Run Linking_With_Guide_3Label.ipynb: 3 label linker model, GISTEmbed Loss (synonym pairings and parent pairings left uncommented)
* Run Linking_With_Guide_7Label.ipynb: 7 label linker model, GISTEmbed Loss (synonym pairings and parent pairings left uncommented)

To evaluate overall linking results, run scoring.py "pred_results.csv" true_res_final.csv
