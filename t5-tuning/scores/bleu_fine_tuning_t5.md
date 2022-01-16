# BLEU Scores for fine tuning T5 small

---

| Challenge/Epochs trained        | 1    | 10   | 25    |
|---------------------------------|------|------|-------|
| Web NLG 2020                    | 13,2 | 17,8 | 19,2  |
| Web NLG 2020 with token inputs  | 14   | 18,9 | 22,7  |
| E2E                             | 8,2  | 52,2 | 54,3  |
| AMR                             | 9,3  | 18,2 | 21,02 |

# BLEU Scores for fine tuning T5 base

| Challenge/Epochs trained        | 1    | 10   | 25    |
|---------------------------------|------|------|-------|
| Web NLG 2020                    | 19,7 | 23,1 | 23,8  |
| Web NLG 2020 with token inputs  | 14,7 | 17,7 | 20.7  |
| E2E                             | 52,0 | 54,8 |  55,6 |
| AMR                             | 14,5 | 19,5 |  17,8 |

##Example Language Generation in AMR by T5 base (1 Epoch):

## Predictions:

[' I had been waiting for a long time... "      ',

' I could see him revived little by little, as he resusc', 

' " Men, dear men, men, men, men, men', 

' I said to him : " I am afraid of him, "   ',

' He was certain, but he was afraid.      ']

##References:

[['I waited a long time .'],

['I could see that he was reviving little by little .'], 

['" Dear little man , "'], 

['I said to him , " you are afraid ... "'], 

['He was afraid , there was no doubt about that .']]
{
---

# BLEU Scores for fine tuning T5 large

---