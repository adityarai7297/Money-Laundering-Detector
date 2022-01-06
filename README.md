# Money-Laundering-Detector

This program checks for common money laundering patterns such as "Smurfing", in a payment transaction network spanning over a few days and/or months.
The models used for detection were trained on a dataset of 24 million financial transactions of an African mobile company.
We used gradient-boosted Decision Trees, and heiranchical clustering to classify fraudulent transactions, reaching a precision of 91%, when identifying subnetworks of fraudulant transactions.

**Smurfing** is a common placement technique. Cash from illegal sources is divided between 'deposit specialists' or 'smurfs' who make multiple deposits into multiple accounts (often using various aliases) at any number of financial institutions. In this way, money enters the financial system and is then available for layering. Suspicion is often avoided as it is difficult to detect any connection between the smurfs, deposits and accounts.

![images](https://user-images.githubusercontent.com/24310341/148461267-92ecfa97-b3dc-47df-adc9-3fda3b05b5db.jpg)



**Structuring** involves splitting transactions into separate amounts under AUD10,000 to avoid the transaction reporting requirements of the FTR Act and AML/CTF Act. Many money launderers rely on this placement technique because numerous deposits can be made without triggering the cash reporting requirements. However, it can backfire if an attentive financial institution notices a pattern of deposits just under the reportable threshold. This can lead to reporting such activity to AUSTRAC under the suspicious activity provisions of these instruments. Structuring is a criminal offence itself, as well as an indicator of other potentially illegal activity.

![x](https://user-images.githubusercontent.com/24310341/148461289-632b086a-d653-4b98-82e7-4a27f38c6301.png)

