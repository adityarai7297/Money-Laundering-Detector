# Money-Laundering-Detector

This program checks for common money laundering patterns such as "Smurfing", in a payment transaction network spanning over a few days and/or months.
The models used for detection were trained on a dataset of 24 million financial transactions of an African mobile company.
We used gradient-boosted Decision Trees, and heiranchical clustering to classify fraudulent transactions, reaching a precision of 91%, when identifying subnetworks of fraudulant transactions.

