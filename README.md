
# Lottery numbers with Naive Bayes

Trying to prove to my father that lottery numbers cannot be predicted and that there is no pattern to be found.

This approach predicts and calculates the best next fit for each pick individually, by looking at the last numbers and checking which number fits best next.  
The learning process always looks at `ATTRIBUTES` numbers as attributes, with the `ATTRIBUTES+1`th number being the class.

So for example the following numbers
`3-6-30-31-32-43,   
2-22-24-33-37-46,  
6-7-12-17-18-34,
14-16-21-28-31-39`

with `ATTRIBUTES =  2` would give these training vectors for the __first__ pick:

`[3, 2, 6], [2, 6, 14]`